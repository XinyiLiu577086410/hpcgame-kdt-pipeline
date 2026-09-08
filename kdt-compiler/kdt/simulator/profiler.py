"""
Perfetto / Chrome Trace Event Format profiler for the KDT-DSL simulator.

The simulator executes jobs one at a time and records, for every instruction,
the cycle at which it was issued and the cycle at which it finished.  This
module turns those records into a JSON file that can be opened directly in
https://ui.perfetto.dev (or chrome://tracing).

Trace layout
------------
* One ``pid`` per SM (``SM0``, ``SM1``, ...).
* One ``tid`` per hardware unit inside an SM:

  - ``JOB``  : whole-job span (one slice per job)
  - ``ISU``  : one 1-cycle slice per issued instruction (issue slot)
  - ``MEM``  : global <-> SPM load / store latency window
  - ``VXM``  : vector unit execution window
  - ``MXM``  : matrix unit execution window
  - ``CTRL`` : zero-latency control instructions (copy / fill / where)

Every instruction slice carries the issue-cycle constraint breakdown in its
``args`` (``isu`` / ``queue`` / ``dep_write`` / ``dep_read``), which makes it
easy to see *why* a slice could not start earlier.
"""

import json
from typing import Any, Dict, List, Optional, Sequence

# Lane (thread) ids used inside every SM track.
UNIT_TID: Dict[str, int] = {
    "JOB": 0,
    "ISU": 1,
    "MEM": 2,
    "VXM": 3,
    "MXM": 4,
    "CTRL": 5,
}


class TraceRecorder:
    """Collects instruction events during a simulation run.

    All cycles stored here are *local* to a job (starting at 0).  The global
    timestamps are computed in :func:`write_perfetto_trace` once the jobs have
    been assigned to SMs.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record(
        self,
        job_id: int,
        unit: str,
        start: int,
        end: int,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        update_isu: bool = False,
    ) -> None:
        self.events.append(
            {
                "job": int(job_id),
                "unit": unit,
                "start": int(start),
                "end": int(end),
                "name": name,
                "args": args or {},
                "update_isu": bool(update_isu),
            }
        )

    def __len__(self) -> int:
        return len(self.events)


def write_perfetto_trace(
    events: Sequence[Dict[str, Any]],
    job_sm: Sequence[int],
    job_start: Sequence[int],
    job_cycles: Sequence[int],
    path: str,
    num_sms: int,
) -> None:
    """Serialize recorded events into a Chrome Trace Event JSON file.

    Args:
        events: events produced by :class:`TraceRecorder`.
        job_sm: SM index assigned to each job.
        job_start: first cycle of each job on its SM.
        job_cycles: duration (in cycles) of each job.
        path: output ``.json`` path.
        num_sms: number of SMs (used to emit the process/thread metadata).
    """
    trace_events: List[Dict[str, Any]] = []

    for sm in range(num_sms):
        trace_events.append(
            {"name": "process_name", "ph": "M", "pid": sm, "args": {"name": f"SM{sm}"}}
        )
        for unit, tid in UNIT_TID.items():
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": sm,
                    "tid": tid,
                    "args": {"name": unit},
                }
            )

    for event in events:
        job = event["job"]
        offset = job_start[job]
        sm = job_sm[job]
        unit = event["unit"]
        tid = UNIT_TID.get(unit, 0)
        duration = max(0, event["end"] - event["start"])
        args = dict(event["args"])
        args["job"] = job

        trace_events.append(
            {
                "name": event["name"],
                "cat": unit,
                "ph": "X",
                "ts": offset + event["start"],
                "dur": duration,
                "pid": sm,
                "tid": tid,
                "args": args,
            }
        )

        if event["update_isu"]:
            trace_events.append(
                {
                    "name": event["name"],
                    "cat": "ISU",
                    "ph": "X",
                    "ts": offset + event["start"],
                    "dur": 1,
                    "pid": sm,
                    "tid": UNIT_TID["ISU"],
                    "args": {"job": job},
                }
            )

    for job in range(len(job_sm)):
        trace_events.append(
            {
                "name": f"job{job}",
                "cat": "JOB",
                "ph": "X",
                "ts": job_start[job],
                "dur": job_cycles[job],
                "pid": job_sm[job],
                "tid": UNIT_TID["JOB"],
                "args": {"job": job},
            }
        )

    with open(path, "w") as f:
        json.dump(
            {
                "traceEvents": trace_events,
                "displayTimeUnit": "ns",
                "metadata": {
                    "source": "kdt-simulator",
                    "time_unit": "cycles",
                },
            },
            f,
        )
