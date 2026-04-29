# 算子设计初探（kernel-design-trial）

## 背景

想接触 CUDA 算子设计，却苦于 CuTe / CUTLASS 太难写而不会编写？想精确控制算子内部的时序和流水线，却因为 Triton / cuTile 等语言的抽象层次过高而无法实现？想了解 GPU 硬件架构，却苦于没有实战经验？现在，你的机会来了！

无需掌握复杂的 CUDA C++ 语法，无需了解 CuTe / CuTe DSL 中复杂的 layout 格式。在这道题目中，你将使用我们设计的一个高度简化的算子设计语言（名叫 kernel-design-trail DSL，简称 KDT-DSL），以“数据块”（Tile）为粒度，设计逻辑、排布流水，最终编写属于你的高性能算子！

## 题目总览

由于这道题目的题面比较长（不过别怕，主要的篇幅都在描述那个 KDT-DSL 该怎么写），所以我们先来一个总览。整个题目大概分为这几个部分：

- 我们将首先介绍 KDT-DSL 的编程模型。看完了这部分内容，你应该有能力写出正确的 KDT-DSL 代码。
- 接下来，我们会介绍我们所使用的硬件（称为 KDT-TPU, Kernel-Design-Trail Tile Processing Unit）的内部结构、执行模型、调度逻辑、以及各个指令单元的吞吐。看完了这部分内容，你就可以对你编写的 KDT-DSL 代码进行调优了。
- 接下来，我们将给出具体的任务要求，包括输入输出规格、性能目标等。任务类似于“实现一个计算 $\vec c = \vec a + \vec b$ 的算子”或者“实现一个矩阵乘法”。你需要使用 KDT-DSL 编写对应的算子，并且达到一定的性能指标。
- 编写好算子之后，你可以使用我们提供的模拟器来测试和验证你的设计。模拟器会做以下几件事情：
    1. 功能验证：使用 CPU 模拟你的算子的执行结果，确保你的算子在功能上是正确的。
    2. 性能评估：使用 CPU 模拟你的算子在 KDT-TPU 上的执行时序，计算你的算子的吞吐量、延迟等性能指标。
    3. 根据上述两个指标打分。
- 准备好之后，就可以提交你的设计了！我们会根据你的设计进行评测，并给出反馈。理论上评测机返回的得分应该和你本地模拟器跑出来的结果是一致的。

## 编程模型

这一章会大体概括 KDT-DSL 的编程模型。看完了这部分内容，你应该有能力写出正确的 KDT-DSL 代码。

KDT-TPU 大体上分为两个组件：SM（Streaming Multiprocessor，流式多处理器）和显存（Global Memory）。显存负责存储全局的输出、输出数据，而 SM 则负责执行计算任务。SM 内部的有一定量的片上存储（Scratchpad memory, SPM）来存储临时的计算结果。一个 KDT-TPU 上只有一个显存，但可以包含多个 SM，每个 SM 独立运作。

每一个使用 KDT-DSL 编写的 Kernel 在运行时都会启动指定个数个 Job，每个 Job 会被分配到某一个 SM 上执行，每个 SM 每一时刻最多只能执行一个 Job。如果有多个 Job 被分配到同一个 SM 上，那么这些 Job 会被顺序执行。因此，不同 Job 之间可能会串行执行，也可能会并行执行。不同 Job 之间独立执行，互不干扰。

因此，从最外层来看，KDT-DSL 包含两个部分：对于 Job 内部计算细节的描述，以及应该启动多少个 Job。其接口如下：

```python
import kdt

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    # 计算应该启动多少个 job 并返回
    num_jobs = task_args['N'] / 1024
    return num_jobs

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def my_kernel(task_args: dict[str, int], io_tensors: dict[str, kdt.Tile]):
    # 这里是 job 内部的计算逻辑
    ...
```

其中，`task_args` 包含与问题规模（比如矩阵乘法任务中的矩阵大小），其格式视具体任务而定，详见后文“测试点与评分标准”一章。

你可以使用以下语法来编译并启动一个 KDT-DSL Kernel：

```python
my_kernel_compiled = my_kernel.compile(task_args, io_tensors)
kdt.launch_kernel(my_kernel_compiled, io_tensors)
```

其中，`io_tensors` 为存储了输入和输出张量的字典，其 key 为张量名称，value 在调用时为对应的 PyTorch 张量对象（`torch.Tensor`），在 Kernel 执行时为 `kdt.Tile`。这些张量均需要位于 CPU 上。

在 Job 的计算逻辑内部，你可以调用各种 KDT-DSL 指令来描述计算和数据传输。KDT-DSL 是一种面向数据块（Tile）的编程语言，在 KDT-DSL 中，所有的数据均以“数据块”（Tile）的形式进行处理。每个数据块都是一个张量（Tensor，可以理解为“多维数组”），包含若干个元素。数据块的形状（shape）与数据类型（dtype）均可以自定义。

每个数据块要么位于全局显存上，要么位于 SM 内部的 SPM 上。算子的输入输出数据块均位于显存上，而中间计算过程中产生的临时数据则可以暂时存储在 SPM 上面。由于 KDT-DSL 中所有的计算指令的输入、输出都必须是 SPM 上的数据块（而不能是显存上的数据块），因此在进行计算之前，必须先将显存上的数据块搬运到 SPM 上；计算完成之后，如果需要将结果保存回显存，也需要将 SPM 上的数据块搬运回显存。

因此，从一个高层次视角上来看，一个 KDT-DSL 算子的大致流程如下：

- 使用 `kdt.alloc_spm` 指令在 SPM 上分配若干个数据块，用于存储中间结果。
- 使用 `kdt.load` 指令将显存上的输入数据块搬运到 SPM 上。
- 使用各种计算指令（比如 `kdt.add`, `kdt.matmul` 等）对 SPM 上的数据块进行计算，产生中间结果。
- 使用 `kdt.store` 指令将 SPM 上的结果数据块搬运回显存。

由于 SPM 的容量有限，因此可能需要对输入数据进行分块处理，将输入数据切成小块并逐块搬运和计算。

为了简化题目，KDT-DSL 中的数据块仅支持两种数据类型：

- 32 为浮点数（`float32`）。这个浮点数等价于 C++ 中的 float 类型，遵循 [IEEE-754 标准](https://en.wikipedia.org/wiki/IEEE_754)。
- 布尔类型（`bool`）

## KDT-DSL 介绍

### 元信息获取指令

#### `kdt.get_job_id`

```python
kdt.get_job_id() -> int
```

获取当前 Job 的 ID。Job ID 是一个从 0 开始的整数，范围为 `[0, num_jobs)`，其中 `num_jobs` 为启动的 Job 总数。

### 数据块创建指令

#### `kdt.alloc_spm`

```python
kdt.alloc_spm(shape: tuple[int, ...], dtype: str = "float32", name: str = "", init_value = 0) -> Tile
```

在 SPM 中分配一个数据块，数据块的标签为 `name`，其形状为 `shape`。如果提供了 `init_value`，则将数据块中的所有元素初始化为 `init_value`。其占用的 SPM 空间大小为 `shape` 中各维度乘积再乘以每个元素的大小（每个 float32 占用 4 字节，每个 bool 占用 $1/8$ 字节）。返回值为一个数据块对象（Tile）。

`kdt.alloc_spm` 只能出现在一个 KDT-DSL kernel 的顶层，也即，不能出现在任何控制流语句（比如循环）内部。

示例：

```python
tile = kdt.alloc_spm((16, 16), dtype='float32', init_value=1.0) # 分配一个 16x16 的数据块，并将其初始化为 1.0。该数据块占用 16*16*4 = 1024 字节的 SPM 空间。
tile = kdt.alloc_spm((32, ), dtype='bool') # 分配一个包含 32 个布尔值的一维数据块。该数据块占用 32*(1/8) = 4 字节的 SPM 空间。
```

<!-- #### `kdt.get_global_tile`

```python
kdt.get_global_tile(name: str) -> Tile
```

获取显存上名为 `name` 的数据块。返回值为一个数据块对象（Tile）。该数据块的形状和数据类型由输入输出张量决定。

`name` 需要是 `input_tensors` 或 `output_tensors` 中存在的张量名称。 -->

### 形状变换指令

#### `kdt.squeeze`

```python
kdt.squeeze(x: Tile, dim: int) -> Tile
```

删除数据块 `x` 在维度 `dim` 上的大小为 1 的轴。返回值为一个新的数据块，形状为删除后的形状。

你需要保证 `x` 在 `dim` 维度上的大小确实为 1，否则行为未定义，编译器可能不会报错，但其运行结果可能不正确。

生成的 Tile 不占用任何额外的 SPM 空间，只是对原数据块的一种新的视图（view）。

示例：

```python
x = kdt.alloc_spm((2, 3, 1))
# 这里假设 x 的内容为 [[[1], [2], [3]], [[4], [5], [6]]]
y = kdt.squeeze(x, dim=2)
# y 的内容为 [[1, 2, 3], [4, 5, 6]]
```

#### `kdt.unsqueeze`

```python
kdt.unsqueeze(x: Tile, dim: int) -> Tile
```

在数据块 `x` 的维度 `dim` 上插入一个大小为 1 的轴。返回值为一个新的数据块，形状为插入后的形状。

生成的 Tile 不占用任何额外的 SPM 空间，只是对原数据块的一种新的视图（view）。

示例：

```python
x = kdt.alloc_spm((2, 3))
# 这里假设 x 的内容为 [[1, 2, 3], [4, 5, 6]]
y = kdt.unsqueeze(x, dim=2)
# y 的内容为 [[[1], [2], [3]], [[4], [5], [6]]]
```

#### `kdt.broadcast_to`

```python
kdt.broadcast_to(x: Tile, dim: int, new_size: int) -> Tile
```

将数据块 `x` 在维度 `dim` 上进行广播（broadcast）操作，该维度的原始大小需要为 1，广播后该维度大小为 `new_size`。

对大小不为 1 的维度进行 broadcast 是未定义行为，编译器可能不会报错，但其运行结果可能不正确。

生成的 Tile 不占用任何额外的 SPM 空间，只是对原数据块的一种新的视图（view）。

示例：

```python
x = kdt.alloc_spm((1, 3))  # 分配一个 1x3 的数据块
# 这里假设 x 的内容为 [[1, 2, 3]]
y = kdt.broadcast_to(x, 0, 4)  # 将 x 广播为 4x3 的数据块
# 现在 y 的的内容为 [[1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3]]
```

#### `kdt.transpose`

```python
kdt.transpose(x: Tile, dim0: int, dim1: int) -> Tile
```

交换数据块 `x` 在 `dim0` 和 `dim1` 维度上的数据。返回值为一个新的数据块，形状为交换后的形状。

生成的 Tile 不占用任何额外的 SPM 空间，只是对原数据块的一种新的视图（view）。

示例：

```python
x = kdt.alloc_spm((2, 3))  # 分配一个 2x3 的数据块
# 这里假设 x 的内容为 [[1, 2, 3], [4, 5, 6]]
y = kdt.transpose(x, 0, 1)  # 交换第 0 和第 1 维度
# 现在 y 的内容为 [[1, 4],
#                  [2, 5],
#                  [3, 6]]
```

#### `kdt.slice`

```python
kdt.slice(x: Tile, dim: int, start: int, end: int) -> Tile
```

对数据块 `x` 在指定的 `dim` 维度上进行切片，返回从 `start`（包含）到 `end`（不包含）之间的子数据块。返回值为一个新的数据块，形状为切片后的形状。

生成的 Tile 不占用任何额外的 SPM 空间，只是对原数据块的一种新的视图（view）。

示例：

```python
x = kdt.alloc_spm((4, 3))  # 分配一个 4x3 的数据块
# 这里假设 x 的内容为 [[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9],
#                  [10, 11, 12]]
y = kdt.slice(x, dim=0, start=1, end=3)  # 对 x 在第 0 维度上切片，取第 1 行到第 3 行
# 现在 y 的内容为 [[4, 5, 6],
#                  [7, 8, 9]]
```

**KDT-DSL 支持 Python 的列表切片（也即，类似 `x[:, 1:10, 3]` 这种写法）**，比如

```python
x = kdt.alloc_spm((20, 20, 20))
y = x[:, 1:10, 3]
```

### 计算指令

对于所有计算指令，输入和输出均为 SPM 上的数据块；并且，除了 `matmul` 指令之外，所有计算指令的输入、输出数据块的形状必须完全相同。

#### `kdt.exp`

```python
kdt.exp(x: Tile, out: Tile, y: float)
```

逐元素计算 $out = y^{x}$，其中 `y` 必须大于 0。

#### `kdt.log`

```python
kdt.log(x: Tile, out: Tile, y: float)
```

逐元素计算 $out = \log_{y}(x)$，其中 `y` 必须大于 0 且不等于 1。

#### `kdt.pow`

```python
kdt.pow(x: Tile, out: Tile, y: float)
```
逐元素计算 $out = x^{y}$。

#### `kdt.add` / `kdt.sub` / `kdt.mul` / `kdt.div` / `kdt.fma`

```python
kdt.add(a: Tile, b: Tile, out: Tile)
kdt.sub(a: Tile, b: Tile, out: Tile)
kdt.mul(a: Tile, b: Tile, out: Tile)
kdt.div(a: Tile, b: Tile, out: Tile)
kdt.fma(a: Tile, b: Tile, c: Tile, out: Tile)
```

分别逐元素计算 $out = a + b$，$out = a - b$，$out = a * b$，$out = a / b$，以及 $out = a * b + c$。

#### `kdt.max` / `kdt.min`

```python
kdt.max(a: Tile, b: Tile, out: Tile)
kdt.min(a: Tile, b: Tile, out: Tile)
```

计算 $out = \max(a, b)$ 或 $out = \min(a, b)$。

#### `kdt.logical_and` / `kdt.logical_or`

```python
kdt.logical_and(a: Tile, b: Tile, out: Tile)
kdt.logical_or(a: Tile, b: Tile, out: Tile)
```

分别逐元素计算 $out = a \operatorname{and} b$ 和 $out = a \operatorname{or} b$。`a`，`b` 和 `out` 均必须为布尔类型的数据块。

#### `kdt.le` / `kdt.leq` / `kdt.eq` / `kdt.neq`

```python
kdt.le(a: Tile, b: Tile, out: Tile)
kdt.leq(a: Tile, b: Tile, out: Tile)
kdt.eq(a: Tile, b: Tile, out: Tile)
kdt.neq(a: Tile, b: Tile, out: Tile)
```

分别逐元素计算 $out = (a < b)$，$out = (a \leq b)$，$out = (a == b)$，以及 $out = (a \neq b)$。`out` 必须为布尔类型的数据块，`a` 和 `b` 必须为浮点类型的数据块。

#### `kdt.matmul`

```python
kdt.matmul(a: Tile, b: Tile, out: Tile, accumulate: bool = False)
```

计算矩阵乘法 $out = a \times b$。假设 `a` 的形状为 `(M, K)`，`b` 的形状为 `(K, N)`，那么 `out` 的形状必须为 `(M, N)`。

如果 `accumulate` 参数为 True，则计算 $out = out + a \times b$。

#### `kdt.reduce`

```python
kdt.reduce(x: Tile, dim: int, op: str, out: Tile)
```

对数据块 `x` 在指定的 `dim` 维度上进行归约（reduce）操作，归约操作由参数 `op` 指定，可以是以下几种：

- `'sum'`：求和
- `'max'`：求最大值
- `'min'`：求最小值

`out` 应为一个数据块，其形状为 `x` 的形状中去掉 `dim` 维度后的形状。

示例：

```python
x = kdt.alloc_spm((2, 3))  # 分配一个 2x3 的数据块
# 这里假设 x 的内容为 [[1, 2, 3], [4, 5, 6]]

y = kdt.alloc_spm((3, ))  # 分配一个 3 元素的一维数据块用于存储结果
kdt.reduce(x, dim=0, op='sum', out=y)  # 对 x 在第 0 维度上求和
# 现在 y 的内容为 [5, 7, 9]

z = kdt.alloc_spm((2, ))  # 分配一个 2 元素的一维数据块用于存储结果
kdt.reduce(x, dim=1, op='max', out=z)  # 对 x 在第 1 维度上求最大值
# 现在 z 的内容为 [3, 6]
```

### 选择指令

#### `kdt.where`

```python
kdt.where(cond: Tile, a: Tile, b: Tile, out: Tile)
```

根据条件数据块 `cond` 的值，从数据块 `a` 和 `b` 中选择元素填充到 `out` 中。具体来说，对于 `cond` 中的每个元素：
- 如果该元素为 True，则将对应位置的 `a` 中的元素赋值给 `out`。
- 如果该元素为 False，则将对应位置的 `b` 中的元素赋值给 `out`。

示例：

```python
cond = kdt.alloc_spm((3, ))  # 分配一个 3 元素的一维布尔数据块
# 假设 cond 的内容为 [True, False, True]
x = kdt.alloc_spm((3, ))  # 分配一个 3 元素的一维数据块
# 假设 x 的内容为 [10, 20, 30]
y = kdt.alloc_spm((3, ))  # 分配一个 3 元素的一维数据块
# 假设 y 的内容为 [1, 2, 3]
out = kdt.alloc_spm((3, ))  # 分配一个 3 元素的一维数据块用于存储结果
kdt.where(cond, x, y, out)  # 根据 cond 从 x 和 y 中选择元素填充到 out 中
# 现在 out 的内容为 [10, 2, 30]
```

#### `kdt.copy`

```python
kdt.copy(src: Tile, dst: Tile)
```

将数据块 `src` 中的内容逐元素复制到数据块 `dst` 中。`src` 和 `dst` 的形状、数据类型必须完全相同。

请注意，如果 `a` 和 `b` 是两个数据块，那么 `a = b` 这条语句并不会将 `b` 的内容复制到 `a` 中，而是将 `a` 重新绑定（rebind）到 `b` 所指向的数据块上。因此，如果你想要将数据块 `b` 的内容复制到数据块 `a` 中，请使用 `kdt.copy(b, a)`。

#### `kdt.fill`

```python
kdt.fill(x: Tile, value: float | bool)
```

将数据块 `x` 中的所有元素填充为指定的 `value`。

### 数据传输指令

数据传输指令包括两个：`kdt.load`（将数据从显存搬运至 SM 内部的 SPM），以及 `kdt.store`（将数据从 SPM 搬运至显存）。

对于数据传输指令，其输入数据块和输出数据块的形状、数据类型必须完全相同。

#### `kdt.load`

```python
kdt.load(src: Tile, dst: Tile)
```

`kdt.load` 和 `kdt.store` 都不支持越界访问（out-of-bound access），在越界时会提示 IndexError 错误。

将显存上的数据块 `src` 搬运到 SPM 上的数据块 `dst` 中。

#### `kdt.store`

```python
kdt.store(src: Tile, dst: Tile)
```

将 SPM 上的数据块 `src` 搬运到显存上的数据块 `dst` 中。

### 调试指令

#### `kdt.print`

```python
kdt.print(x: Union[Tile, ir.Expr], msg: str = "", print_only_if_job0: bool = False)
```

打印调试信息 `msg`，以及数据块（或标量表达式） `x` 中的内容。如果参数 `print_only_if_job0` 为 True，则仅在当前 Job ID 为 0 时打印该信息，否则所有 Job 都会打印该信息。

### 一些限制

- 在 KDT-DSL 的本体中，你只能对 Python 的 `int`/`bool`/`float` 类型、以及 `kdt.Tile` 类型的数据进行操作。此外，你还可以从字典（`dict`）中读取数据（比如 `task_args` 中的数据），但除此之外的操作均不允许（比如不能对字典进行写操作）。
- KDT-DSL 中的所有计算指令的输入都必须是数据块（`kdt.Tile`），而不能是标量（比如 Python 中的 `int`）。如果你确实有让数据块与标量做计算的需求（比如让某个数据块中的数字全部 `+1`），那么可以先使用 `alloc_spm`，在 SPM 上分配一个大小为 1 的数据块，并通过 broadcast 广播至和原数据块一样大，然后与原数据块进行计算。
- 请通过 `import kdt` 来导入 KDT-DSL，而不是 `from kdt import *` 或者 `import kdt as XXX` 这种方式。
- 如果你看到了类似 `InternalError` 的报错，那说明编译器写错了，请联系出题人修复。
- KDT-DSL 中的数据块没有对齐要求。你可以分配任意形状的数据块，KDT-TPU 会自动处理数据块的存储和访问。
- KDT-DSL 支持 Python 的 for 循环（但仅支持 `for var in range(...)` 这种形式的循环），以及 if 语句。但不支持 while 循环、try-except 语句等其他控制流语句，也不支持 multiple assignment（比如 `a, b = kdt.alloc_spm(...), kdt.alloc_spm(...)` 这种写法）与 Python 自带的 `min`, `max` 等函数。
- 由于出题人精力有限，KDT-DSL 仅仅支持 Python 语法的一个子集。如果你看到编译过程中报错 `kdt.language.errors.XXXError`，那么很可能是因为你使用了 KDT-DSL 不支持的 Python 语法。请参考错误信息，修改你的代码。
  ~~饶了我吧，写这玩意的工作量跟一个编译原理的大 lab 一样，我已经献祭了一个周末了，呜呜呜   - intlsy~~
- 现在 KDT-DSL 的前端会对每一个 `if` 和 `for` 均创建一个新的作用域（scope），并且，KDT-DSL 中的标量变量不支持赋值。也就是说，这份代码：
  ```python3
  x = 0
  if True:
    x = 1
  kdt.print(x)
  ```
  相当于在 `if` 内部重新定义了一个新的变量 `x`，而外部的 `x` 并没有被修改，因此最终 `x` 的值仍为 0。这份代码会输出 0。
  为了防止大家写错代码，KDT-DSL 默认情况下会禁止在 `if` 或 `for` 语句内部创建与外部同名的变量（也即，不允许 variable shadowing）。你可以通过定义 `KDT_ENABLE_VARIABLE_SHADOWING=1` 来允许这种行为。
  为了兼容先前的代码，线上评测系统默认开启这个选项。


## KDT-DSL 例子

下面是一个计算 $\vec c = \vec a + \vec b$ 的 KDT-DSL 算子的例子：

```python
from typing import Dict

import kdt
import torch

def calculate_num_jobs(task_args: dict[str, int]) -> int:
    BLOCK_SIZE = 4
    vec_size = task_args['vec_size']
    assert vec_size % BLOCK_SIZE == 0, "vec_size 必须是 BLOCK_SIZE 的整数倍"
    num_jobs = vec_size // BLOCK_SIZE  # 每个 job 处理 BLOCK_SIZE 个元素，因此一共需要启动 vec_size / BLOCK_SIZE 个 job
    return num_jobs

@kdt.kernel(num_jobs_calculator=calculate_num_jobs)
def vector_add_kernel(task_args: Dict[str, int], io_tensors: Dict[str, kdt.Tile]):
    BLOCK_SIZE = 4   # 目前 kdt.kernel 还不能用 global 变量，所以这里重新定义一遍
    vec_size = task_args['vec_size']
    job_id = kdt.get_job_id()  # 获取当前 job 的 ID
    start_idx = job_id * BLOCK_SIZE  # 计算当前 job 处理的起始索引
    end_idx = start_idx + BLOCK_SIZE  # 计算当前 job 处理的结束索引

    # 分配 SPM 上的数据块
    a_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    b_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')
    c_tile = kdt.alloc_spm((BLOCK_SIZE,), dtype='float32')

    # 加载输入数据到 SPM
    kdt.load(io_tensors['a'][start_idx: end_idx], a_tile)
    kdt.load(io_tensors['b'][start_idx: end_idx], b_tile)

    # 执行向量加法
    kdt.add(a_tile, b_tile, c_tile)

    # 存储结果回显存
    kdt.store(c_tile, io_tensors['c'][start_idx: end_idx])

def main():
    vec_size = 16  # 向量大小
    a = torch.randn(vec_size, dtype=torch.float32)  # 输入向量 a
    b = torch.randn(vec_size, dtype=torch.float32)  # 输入向量 b
    c = torch.empty(vec_size, dtype=torch.float32)  # 输出向量 c

    task_args = {'vec_size': vec_size}
    io_tensors = {'a': a, 'b': b, 'c': c}

    compiled_kernel = vector_add_kernel.compile(task_args, io_tensors)
    # compiled_kernel.print_ir()  # 打印中间表示（IR）

    tpu_spec = kdt.TPUSpec(num_sms=1, load_store_latency=100)
    num_cycles = kdt.launch_kernel(compiled_kernel, io_tensors, tpu_spec)

    print("Result c:", c)
    assert torch.allclose(c, a + b), "结果不正确！"
    print(f"Kernel executed in {num_cycles} cycles.")

if __name__ == '__main__':
    main()
```

（这个例子可以在测试点 1 中拿到一部分正确性分数，但性能很差）

## GPU 体系结构建模

这一章中，我们将介绍一下我们所使用的 GPU（下文称为 KDT-TPU）体系结构模型。这个模型是一个高度简化的版本，主要目的是让你能够专注于算子设计，而不需要过多关注底层硬件细节。看完了这一章之后，你应该能够理解 KDT-DSL 代码在 KDT-TPU 上的执行过程、分析算子的性能瓶颈、并据此优化你的设计。

### 内部结构

KDT-TPU 包含两个关键组件：SM（可以理解为某种意义上的“计算核心”）和显存。其中，SM 负责发射各种指令，进行计算任务，而显存则负责存储输入、输出数据。SM 和显存之间通过一个高带宽的总线连接，这个总线的带宽、以及显存的带宽均是无限的，但每次传输的延迟均为 $L_{mem}$ 个时钟周期，其中 $L_{mem}$ 是一个预定义的常数，在不同测试点中不同。这意味着，当 SM 发起一次内存访问请求后，需要等待 $L_{mem}$ 个时钟周期才能收到数据。

上文我们提到，每个 KDT-DSL 的 Job 都会被分配到某一个 SM 上执行，不同 Job 之间互不干扰。每个 SM 内部包含四个单元：向量计算单元（VXM）、矩阵乘法计算单元（MXM）、片上存储（Scratchpad memory, SPM）以及指令发射调度单元（Issue & scheduling unit, ISU）。其中，MXM 能且仅能负责计算矩阵乘法指令 `kdt.matmul`，而 VXM 则负责其他计算指令（包括向量加法 `kdt.add`、向量乘法 `kdt.mul`等）。MXM 和 VXM 的输入、输出均为 SPM，访存指令则会在 SPM 和显存之间来回搬运数据。整个 KDT-TPU 包含若干 SM，不同 SM 之间是完全独立的，互不干扰。每个 SM 都有自己的 MXM, VXM, SPM 和 ISU。

SM 内部的 SPM 大小有限，具体大小见“测试点与评分标准”一章中的说明。

### 发射与调度

每个 SM 内部拥有一指令发射调度单元（ISU），负责本 SM 的指令的发射与调度。每个 SM 的 ISU 内部都会维护一个 32 位的特殊寄存器 Program Counter (PC)，初始值为 0。在运行时，ISU 会反复进行以下工作：

- 读取并解析当前 PC 指向的 KDT-DSL 指令。
- 如果元信息获取指令、数据块创建指令、形状变换指令、调试指令选择指令等不涉及计算的指令，ISU 会直接执行这些指令，并立即让 PC 指向下一条指令。ISU 在这条指令上不会消耗任何周期。
- 如果这条指令是循环指令，则根据指令的语义修改 PC 的值，从而实现循环。ISU 在这条指令上不会消耗任何周期。
- 如果这条指令是计算指令（比如计算两个小块的矩阵乘法），则会等待这条指令所依赖的指令全部完成（见下文），然后将这条指令发射到对应的计算单元（MXM 或 VXM），并让 PC 指向下一条指令。这需要消耗一周期。
- 如果这条指令是访存指令（`kdt.load` 或 `kdt.store`），则会等待这条指令所依赖的指令全部完成（见下文），然后发起访存请求，并让 PC 指向下一条指令。这需要消耗一周期。

ISU 每周期最多只能发射一条计算指令或访存指令。

KDT-TPU 遵循着“顺序发射，乱序执行”的逻辑，也即，只要一条指令不依赖于之前的指令的数据，那么它就可以立刻发射，无需等待之前的指令执行完成；如果一条指令依赖于之前的某条指令的输出数据，那么它需要等待被依赖的指令执行完成后才能发射，后面的指令也不能先于这条指令发射。

依赖包括三种情况：
- 读后写（Read After Write, RAW）：某条指令需要读取之前某条指令的输出数据。
- 写后读（Write After Read, WAR）：某条指令需要写入之前某条指令正在读取的数据块。
- 写后写（Write After Write, WAW）：某条指令需要写入之前某条指令正在写入的数据块。

KDT-TPU 与编译器会自动帮你分析指令之间的依赖关系，并在必要的时候等待被依赖的指令执行完成。此处“分析指令之间的依赖关系”的粒度是“单个元素”而不是“整个数据块”。举个例子：假设 `a` 是一个大小为 `[8, 8]` 的数据块，其中 `a[0:4, :]` 将在第 10 周期就绪，`a[4:8, :]` 将在第 20 周期就绪，那么一条以 `a[1, :]` 作为输入的指令将在第 10 周期就绪后就可以发射，而不需要等到第 20 周期。

MXM 和 VXM 都有自己的指令队列，在发射计算指令时，ISU 会将这条计算指令放入 MXM 或 VXM 的指令队列中。MXM 和 VXM 会按照“先进先出”的顺序，依次处理自己的指令队列中的每条指令。如果前面的指令还没有执行完成，那么后面的指令就需要等待。计算指令中的 `matmul` 指令会被发射到 MXM 中，而其他计算指令（比如 `add`, `mul` 等）则会被发射到 VXM 中。特别地，如果开启了 `matmul` 指令中的 `accumulate` 参数，那么该指令仍然会被发射到 MXM 中，且 accumulate 参数不会影响指令的执行周期。

### 指令单元吞吐

VXM 单元每周期可以处理 128 个 float32 元素的计算任务，也即，消耗的周期为输入的数据块中的数据个数 $/128$。MXM 单元每周期可以处理 2048 个 float32 元素的矩阵乘法计算任务，但需要把计算块的各个维度大小 pad 至 128 或 64 的倍数（也即，如果输入的 `a` 的形状为 `[M, K]`，`b` 的形状为 `[K, N]`，那么消耗的周期即为 $\lceil \operatorname{pad}{(M, 128)}\operatorname{pad}{(N, 128)}\operatorname{pad}{(K, 16)} / 2048 \rceil$，也就是说，MXM 单元会把 M, N, K 三个维度分别 pad 到 128, 128, 16 的倍数）。若消耗的周期不是整数，则向上取整。

特别地，“FMA”指令虽然本质上是两个操作（乘法和加法），但在 VXM 中被视为一个整体指令，因此 FMA 指令的吞吐量也是 128 个 float32 元素每周期。这意味着，`kdt.fma(a, b, c, out=out)` 要比 `kdt.add(a, b, out=tmp); kdt.mul(tmp, c, out=out)` 快一倍。

一些例子：

- 对于 `kdt.exp` 指令，如果输入数据块的形状为 `(512,)`，那么该指令需要 $512 / 128 = 4$ 个周期完成。
- 对于 `kdt.add` 指令，如果输入数据块的形状为 `(256,)`，那么该指令需要 $256 / 128 = 2$ 个周期完成。
- 对于 `kdt.matmul` 指令，如果输入数据块 `a` 的形状为 `(64, 32)`，`b` 的形状为 `(32, 128)`，那么该指令需要 $\lceil \frac{\operatorname{pad}(64, 128) \times \operatorname{pad}(32, 16) \times \operatorname{pad}(128, 128)}{2048} \rceil = \lceil 256 \rceil = 256$ 个周期完成。
- 对于 `kdt.fma` 指令，如果输入数据块的形状为 `(300,)`，那么该指令需要 $\lceil 300 / 128 \rceil = 3$ 个周期完成。
- 对于 `kdt.load` 指令，不管输入数据块的形状为何，该指令均需要 $L_{mem}$ 个周期完成。

### 示例

举个例子，对于以下代码（代码前的中括号代表行号）：

```python
# 假设 a, b, c 的大小均为 (16, 128)，d 的大小为 (128, 64)
# 假设 L_mem = 100
[0] kdt.load(io_tensors['a'], a)
[1] kdt.add(a, b, out=out1)
[2] kdt.matmul(a, d, out=out2)
[3] kdt.mul(out1, c, out=out3)
[4] kdt.sub(a[2], b[2], out=out4)
```

那么，KDT-TPU 的执行顺序如下：

- 初始时，PC 指向 [0]
- 第一周期内，发射 [0]，向 SPM 加载数据块 `a`。这条指令需要 $L_{mem} = 100$（备注：不同测试点中的 $L_{mem}$ 不同）个周期完成，因此将在第 $101$ 周期完成。
- 发现 [1] 依赖于 [0] 的数据，因此需要等到 [0] 执行完成，等到第 101 周期，然后发射 [1]，向 VXM 中提交 `out = a+b` 计算任务，将 PC 更新至 [2]。这条指令需要 $16 \times 128/128=16$ 个周期完成，因此将在第 $101+16 = 117$ 周期完成。
- 发现 [2] 不依赖 [1] 的数据，所以无需等待 [1] 完成，直接在第 102 周期时向 MXM 提交计算 `out2 = matmul(a, b)` 的计算任务，并将 PC 更新至 [3]。这条指令需要 $\lceil \operatorname{pad}(16, 128) \times \operatorname{pad}(128, 16) \times \operatorname{pad}(64, 128)/2048 \rceil = 1024$ 个周期完成，因此将在第 $102+1024=1126$ 周期完成。
- 发现 [3] 依赖于 [1] 的输出，因此等待 [1] 执行完成。指令 [1] 在第 117 周期完成，因此在 117 周期时向 VXM 提交 `out3 = out1 * c` 的计算任务，并将 PC 更新至 [4]。这条指令需要 $16 \times 128/128=16$ 个周期完成，因此将在第 $117+16=133$ 周期完成。
- 发现 [4] 不依赖于之前的任何指令的输出，因此在第 118 周期，直接向 VXM 提交 `out4 = a - b` 的计算任务。这条指令需要 $128/128=1$ 个周期完成，但由于 VXM 单元也是顺序执行的，因此这条指令需要等待 [3] 执行完成才能执行。指令 [3] 在第 133 周期完成，因此 [4] 将在第 $133+1=134$ 周期完成。请注意，虽然这条指令中出现了切片（slice）操作，但切片操作并不消耗任何周期，因此其与整体指令的周期数无关。

因此，该 Job 的执行需要消耗 1126 个周期。如果该 Job 在第 1000 周期被调度到某一个 SM 上执行，那么它将在第 1000 + 1126 = 2126 周期完成。整个算子的完成时间为所有 Job 中最晚完成的那个 Job 的完成时间。

请注意，由于 KDT-TPU 是“顺序发射”的，因此哪怕 [4] 不依赖于 [1] 或 [2]，ISU 也不会在发射 [3] 之前就发射 [4]。

### 关于性能分析的提示

除了看算子执行所消耗的周期数量之外，我更建议大家采用“算子的瓶颈资源的利用率”作为性能分析的指标。所谓“瓶颈资源”，是指在算子执行过程中，最繁忙的那个资源（MXM 或 VXM）。所谓“利用率”，是指该资源在整个算子执行过程中，有多少比例的时间是在忙碌地工作，而不是闲置等待。

举个例子，对于矩阵乘法来说，其最忙碌的单元当然是 MXM 单元。一个 SM 的 MXM 单元每周期可以处理 2048 个 float32 元素的矩阵乘法计算任务。如果一个矩阵乘法算子需要计算 $M \times K$ 和 $K \times N$ 两个矩阵的乘法，那么该算子在 MXM 上的理论最少执行周期数为 $\lceil \frac{MNK}{2048} \rceil$。如果该算子实际执行消耗了 $T$ 个周期，那么该算子在 MXM 上的利用率即为 $\frac{\lceil \frac{MNK}{2048} \rceil}{T}$。你可以根据这个指标来分析你的算子设计是否充分利用了 MXM 单元。

这也意味着，某种意义上，“优化算子”的本质就是在优化瓶颈资源的利用率，让其一直满载，同时将使用了其他资源的操作隐藏到瓶颈资源的计算时间中去。

## 测试点与评分标准

本题共有四个任务，每个任务会要求你实现一个特定的算子（比如向量加法、矩阵乘法等）。每个任务会指定输入输出数据的规格（形状、数据类型等），以及性能目标（比如吞吐量、延迟等）。你需要使用 KDT-DSL 编写对应的算子，并且达到指定的性能目标。

- 任务 1：向量加法
- 任务 2：矩阵乘法
- 任务 3：带有细粒度缩放的矩阵乘法
- 任务 4：Flash Attention 算法

每个任务中有若干个测试点，同一个任务的不同测试点之间的计算目标（你要计算什么东西）相同，但输入规模、输出规模、性能目标、KDT-TPU 的硬件指标（包括 $L_{mem}$、SM 数量、每个 SM 的 SPM 大小）等会有多不同。

每个测试点会根据算子的正确性和性能进行评分。每一个测试点都有执行周期要求 $T$，假设你的算子的完成时间为 $T'$，结果是否正确为 $c$（结果正确则 $c=1$，否则 $c=0$），那么你获得的性能分数是 $(\min(1, (T/T')^4) \times 90 + 10) \cdot c$，满分为 100 分。也就是说，只要你的算子的正确性无误，那么即可拿到 10% 的正确性分数，并按比例拿到其余的 90% 的性能分数。本题最终得分为各测试点得分之和。

关于如何提交算子代码，请参考“提交方式”一章。

### 任务 1：向量加法

给定两个大小为 `N` 的向量 `a` 和 `b`，计算它们的和 `c = a + b`，并输出至向量 `c`。

`a` 和 `b` 的大小均为 `[N, ]`，数据类型为 `float32`。保证 $N \bmod 16384 = 0$。

KDT-TPU 的 SM 数量为 1，每个 SM 的 SPM 大小为 128 KB，$L_{mem} = 50$。

测试点规格如下：

| 测试点编号 | 向量大小 N | 执行周期阈值 T |
| ---------- | ---------- | -------------- |
| 1          | $16384$       | 1919810             |
| 2          | $64 \times 16384$       | 11000             |

### 任务 2：矩阵乘法

给定两个矩阵 `A` 和 `B`，计算它们的矩阵乘法乘积 `C = A @ B`，并输出至矩阵 `C`。

`A` 的大小为 `[M, K]`，`B` 的大小为 `[K, N]`，`C` 的大小为 `[M, N]`，数据类型均为 `float32`。保证 $M, N, K$ 均为 512 的整数倍。

KDT-TPU 的 SM 数量为 32，每个 SM 的 SPM 大小为 384 KB，$L_{mem} = 1000$。

测试点规格如下：

| 测试点编号 | M, N, K | 执行周期阈值 T |
| ---------- | ---------- | -------------- |
| 1          | $512, 1024, 2560$       | 23272             |
| 2          | $2048, 512, 3072$       | 54613             |

### 任务 3：带有细粒度缩放的矩阵乘法

现在，你有两个矩阵 `A` 和 `B`，你要计算二者的矩阵乘法结果，并输出至矩阵 `C`。

但，与上一个任务不同的是，`A` 和 `B` 均由两个东西共同表示。以 `A` 为例（其形状应该是 `[M, K]`），我将给你的算子提供一个 base 数组 `Ab`（其形状是 `[M, K]`，类型为 `float32`），以及一个细粒度的 scale factor 数组 `As`（其形状是 `[M, K/64]`）。A 数组中的数字在 K 方向上每 64 个数字共享一个 scale factor，也即，`A[i, j] = Ab[i, j] * As[i, floor(j/64)]`。对于 `B` 来说也是类似的：`B` 的形状是 `[K, N]`，`Bb` 形状为 `[K, N]`，`Bs` 形状为 `[K/64, N]`，`B[i, j] = Bb[i, j] * Bs[floor(i/64), j]`。保证 $M, N, K$ 均为 512 的整数倍。

KDT-TPU 的 SM 数量为 32，每个 SM 的 SPM 大小为 528 KB，$L_{mem} = 1000$。

| 测试点编号 | M, N, K | 执行周期阈值 T |
| ---------- | ---------- | -------------- |
| 1          | $512, 1024, 2560$       | 23272             |
| 2          | $2048, 512, 3072$       | 54613             |

备注：DeepSeek 在其 DeepSeek-V3 大模型中，便十分具有创新性地使用了这种“细粒度缩放”的矩阵乘法，将 `A` 和 `B` 分别量化为 8-bit 浮点数，并使用细粒度的 float32 缩放因子进行缩放，从而在保证模型精度的同时，大幅提升了矩阵乘法的计算效率（因为 8-bit 的矩阵乘法理论上比 16-bit 的快一倍）与存储效率，详见论文 https://arxiv.org/abs/2305.13245 。 DeepSeek 公司也在其开源的 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 仓库中实现了这种矩阵乘法。

### 任务 4：Flash Attention 算法

Attention 算法是现代 Transformer 网络的核心模块，其接受三个矩阵 `Q`（大小 `[S_qo, D]`），`K`（大小 `[S_kv, D]`），`V`（大小 `[S_kv, D]`），并输出矩阵 `O`（大小 `[S_qo, D]`）：

$$O = \operatorname{softmax}(Q K^T, \text{dim}=-1) V$$

*备注：这里我们为了简单，省略了多头注意力、causal mask、softmax scale 等细节。*

正常来说，由于 softmax 操作需要知道所有被 softmax 的数字的大小，我们需要先调用一个矩阵乘法算子，计算出 $Q K^T$ 的值，将其存储到显存上，统计每一行的 softmax 分母，进行 softmax 操作，然后再调用另一个矩阵乘法算子，计算 softmax 的输出（下文称为“Attention score”）与 $V$ 的矩阵乘法。

但是，这样有两个大问题：一是 Attention score 的大小是 `[S_qo, S_kv]` 的，如果训练的上下文较长，那么显存上很有可能存不下；二是这样反复读写 Attention score 会带来极高的额外内存访存开销，大幅拖慢整体的执行速度。

为此，Tri Dao 实验室提出了著名的 Flash Attention 算法，其使用了 online softmax 技巧，打破了 softmax 带来的全局依赖，让 Attention score 不再需要落盘至显存，从而解决了上述两个问题。Flash Attention 一共有三个版本：v1, v2 和 v3。其中，v1 的论文提出了最原始的 Flash Attention 算法，但对应的 CUDA 实现较为粗糙，性能不佳；v2 则是在 v1 的基础上加以改进，提高运行效率；v3 则是根据 NVIDIA 的 Hopper 架构进行了特殊优化，并添加了对低精度（八位浮点小数）的支持。下面是三篇论文的链接：

- v1: https://arxiv.org/abs/2205.14135
- v2: https://arxiv.org/abs/2307.08691
- v3: https://arxiv.org/abs/2407.08608

你的任务便是使用 KDT-DSL 复现上述 Flash Attention 算法，并优化性能。

KDT-TPU 的 SM 数量为 8，每个 SM 的 SPM 大小为 640 KB，$L_{mem} = 1000$。

| 测试点编号 | `S_qo`, `S_kv`, `D` | 执行周期阈值 T |
| ---------- | ---------- | -------------- |
| 1 | $(1024, 128, 128)$ | 8000 |
| 2 | $(1024, 2048, 128)$ | 36408 |
| 3 | $(1024, 4096, 128)$ | 71234 |
| 4 | $(2048, 4096, 128)$ | 142469 |

*注：由于 KDT-DSL 的前端没实现接受 `math.e` 这种 Attribute 作为参数的功能，因此请直接使用 `2.7182818284590451` 而不是 `math.e` 作为 `exp` 的参数*

## 提交与评测

### 提交方式

你需要提交一个 python 文件，在该文件中实现所有任务所需的算子，并实现一个叫做 `get_kernel` 的函数，其签名如下：

```python
def get_kernel(task_id: int) -> kdt.KernelFunction
```

也就是说，它接受一个整数参数 `task_id`，表示任务编号（1 到 5），并返回一个 `kdt.KernelFunction` 对象（使用 `@kdt.kernel` 装饰器装饰的函数）。KDT-DSL 的测试系统会调用该函数，获取对应任务的算子实现，并对其进行编译与测试。

### 评测方式

为了方便你 debug 和测试（也为了防止编译器有潜在的 bug），我们为你提供了一个本地测试框架。线上评测系统使用的评测框架与下发的完全一致。使用该测试框架的流程如下：

- 在你的本地环境中安装 kdt 包（`cd kdt-compiler; pip install -e .`）。
- 编写你的算子代码，并实现 `get_kernel` 函数。假设你的代码文件名为 `my_kdt_kernels.py`。
- 调用 `python3 judger/judger.py --kernel-impl-path my_kdt_kernels.py` 来运行评测系统。该脚本支持 `--task <TASK_ID>` 参数，用于指定只测试某一个任务（`TASK_ID` 为 1 到 4 之间的整数）。如果不指定该参数，则会测试所有任务。该脚本还支持 `--print-ir` 参数，用于打印编译后的中间表示（IR），方便 debug。
- 评测系统会自动编译你的算子代码，并在 KDT-TPU 模拟器上运行你的算子，统计其正确性与性能，并输出最终得分。

单个测试点的模拟执行时间不能超过 20 秒钟，否则该测试点视为失败。

请注意，这是 HPCGame 而不是 GeekGame，比的是大家的 HPC 能力而不是 CTF 能力，因此，所有使用系统漏洞（比如在 python `import` 的时候执行一些奇妙代码）的操作均被禁止。我们会在赛后对提交的代码进行审查，若有发现，该题记为零分。

## 备注

备注：
- 本题目中的 KDT-DSL 是我们专门为这道题目设计的一个简化版的 GPU kernel 设计语言，并不对应任何现有的实际编程语言。
- 本题目中的 TPU 体系结构模型是一个高度简化的版本，简化得甚至不太符合实际的 TPU 设计（比如实际的 TPU 硬件不太可能以元素为粒度分析依赖关系并自动处理数据依赖），仅用于教学和练习目的。
- 为了简化题目，本题目中的数据块只有 `float32` 和 `bool` 两种数据类型。实际的 GPU/TPU 通常会支持多种数据类型，比如 `float16`，`bfloat16`，`float8_e4m3`，甚至 `float4_e2m1` 等等。在某些时候（特别是在计算矩阵乘法的时候），使用更低精度的数据类型可以显著提升性能 —— 这正是 DeepSeek V3 训练成本如此之低的原因之一。
- 如果你真的想涉足 GPU 编程，想像这道题一样精细控制算子内部设计，但又不想接触繁杂的 CUDA / CuTe / CUTLASS，那么建议尝试 [TileLang](https://github.com/tile-ai/tilelang)。
- 如果你感觉自己是干这一行（指算子设计与优化）的料，那么可以考虑本科毕业之后直接去 AI 公司就业写算子。现在就业市场上对能写高性能算子的人才的需求很旺盛，赚钱需趁早，不要读博啦。
