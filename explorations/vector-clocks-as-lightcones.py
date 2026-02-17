#!/usr/bin/env python3
"""
Vector Clocks Are Light Cones
=============================

This script demonstrates a structural analogy between Lamport vector clocks
in distributed systems and light cones in special relativity.

THE ANALOGY:

In special relativity, the light cone of an event E divides all other events
into three regions:
  - Causal past:    events that COULD HAVE influenced E (inside past light cone)
  - Causal future:  events that E COULD influence (inside future light cone)
  - Spacelike:      events with NO causal relation to E — neither can influence
                    the other, and different observers may disagree on their ordering

In distributed systems, vector clocks define the EXACT same structure:
  - If VC(A) < VC(B):  A happens-before B  (A is in B's causal past)
  - If VC(A) > VC(B):  B happens-before A  (A is in B's causal future)
  - If neither:        A and B are concurrent (spacelike separated)

The parallel is not accidental. Both formalisms solve the same problem:
defining causal structure in a system where there is no global clock.

  - In SR: no simultaneity between distant observers (no global time)
  - In distributed systems: no shared clock between processes (no global time)

Both use LOCAL information propagation (light signals / messages) to establish
what CAN be ordered and what CANNOT. The speed of light IS the message latency —
both create a finite propagation horizon for causal influence.

Requirements: Python 3, no external dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Event:
    """An event in our distributed spacetime."""
    process: int
    local_seq: int
    clock: list
    label: str
    kind: str = "internal"  # "internal", "send", "receive"
    partner: Optional[int] = None  # event index of send/receive partner


class DistributedSystem:
    """Simulates distributed processes with vector clocks."""

    def __init__(self, num_processes):
        self.n = num_processes
        self.clocks = [[0] * num_processes for _ in range(num_processes)]
        self.seq = [0] * num_processes
        self.events = []

    def internal_event(self, process, label):
        self.clocks[process][process] += 1
        self.seq[process] += 1
        ev = Event(process, self.seq[process], list(self.clocks[process]), label)
        self.events.append(ev)
        return len(self.events) - 1

    def send_message(self, sender, receiver, send_label, recv_label):
        # Send event
        self.clocks[sender][sender] += 1
        self.seq[sender] += 1
        send_ev = Event(sender, self.seq[sender], list(self.clocks[sender]),
                        send_label, kind="send")
        self.events.append(send_ev)
        send_idx = len(self.events) - 1

        # Receive event: merge clocks
        self.clocks[receiver][receiver] += 1
        for i in range(self.n):
            self.clocks[receiver][i] = max(self.clocks[receiver][i],
                                           self.clocks[sender][i])
        self.seq[receiver] += 1
        recv_ev = Event(receiver, self.seq[receiver], list(self.clocks[receiver]),
                        recv_label, kind="receive")
        self.events.append(recv_ev)
        recv_idx = len(self.events) - 1

        # Link them
        self.events[send_idx].partner = recv_idx
        self.events[recv_idx].partner = send_idx

        return send_idx, recv_idx


def happens_before(vc_a, vc_b):
    """True if a happens-before b (a < b in vector clock ordering)."""
    return all(a <= b for a, b in zip(vc_a, vc_b)) and vc_a != vc_b


def classify(vc_a, vc_b):
    """Classify causal relationship between two events."""
    if vc_a == vc_b:
        return "SAME EVENT"
    if happens_before(vc_a, vc_b):
        return "CAUSAL PAST"      # a is in b's past light cone
    if happens_before(vc_b, vc_a):
        return "CAUSAL FUTURE"    # a is in b's future light cone
    return "SPACELIKE"            # causally independent — no ordering


def draw_spacetime(sys):
    """Draw an ASCII spacetime diagram with causal connections."""
    max_seq = max(e.local_seq for e in sys.events)
    grid_w = 20

    print(f"\n{'':>4s}", end="")
    for p in range(sys.n):
        print(f"{'Process ' + str(p):^{grid_w}s}", end="")
    print()
    print(f"{'':>4s}", end="")
    for p in range(sys.n):
        print(f"{'|':^{grid_w}s}", end="")
    print()

    for t in range(1, max_seq + 1):
        print(f" {t:>2d}  ", end="")
        row_events = {}
        for ev in sys.events:
            if ev.local_seq == t:
                row_events[ev.process] = ev

        for p in range(sys.n):
            if p in row_events:
                ev = row_events[p]
                marker = f"({ev.label})"
                print(f"{marker:^{grid_w}s}", end="")
            else:
                print(f"{'|':^{grid_w}s}", end="")
        print()

        # Show message arrows for send events in this row
        for p in row_events:
            ev = row_events[p]
            if ev.kind == "send" and ev.partner is not None:
                recv = sys.events[ev.partner]
                direction = "--->" if recv.process > ev.process else "<---"
                mid = (ev.process + recv.process) / 2
                print(f"{'':>4s}", end="")
                for pp in range(sys.n):
                    if pp == ev.process and recv.process > ev.process:
                        print(f"{'*--->':^{grid_w}s}", end="")
                    elif pp == ev.process and recv.process < ev.process:
                        print(f"{'<---*':^{grid_w}s}", end="")
                    else:
                        print(f"{'|':^{grid_w}s}", end="")
                print()


def show_light_cone(sys, event_idx):
    """Show the light cone classification for a given event."""
    target = sys.events[event_idx]
    print(f"\n{'=' * 60}")
    print(f"  LIGHT CONE OF EVENT: '{target.label}' (Process {target.process})")
    print(f"  Vector clock: {target.clock}")
    print(f"{'=' * 60}")

    past, future, spacelike = [], [], []

    for i, ev in enumerate(sys.events):
        if i == event_idx:
            continue
        rel = classify(ev.clock, target.clock)
        entry = f"  {ev.label:>12s}  VC={ev.clock}  (P{ev.process})"
        if rel == "CAUSAL PAST":
            past.append(entry)
        elif rel == "CAUSAL FUTURE":
            future.append(entry)
        else:
            spacelike.append(entry)

    print(f"\n  CAUSAL PAST (inside past light cone — could have influenced this event):")
    for e in past:
        print(e)
    if not past:
        print("    (none)")

    print(f"\n  CAUSAL FUTURE (inside future light cone — this event could influence):")
    for e in future:
        print(e)
    if not future:
        print("    (none)")

    print(f"\n  SPACELIKE SEPARATED (outside light cone — no causal ordering):")
    for e in spacelike:
        print(e)
    if not spacelike:
        print("    (none)")


def main():
    print("=" * 60)
    print("  VECTOR CLOCKS ARE LIGHT CONES")
    print("  Causal structure in distributed systems and spacetime")
    print("=" * 60)

    sys = DistributedSystem(3)

    # Build a scenario with interesting causal structure
    a = sys.internal_event(0, "a:init")
    b = sys.internal_event(1, "b:init")
    c = sys.internal_event(2, "c:init")

    # P0 sends to P1 (causal link: a -> d,e)
    d, e = sys.send_message(0, 1, "d:send", "e:recv")

    # P2 does independent work (spacelike to d,e)
    f = sys.internal_event(2, "f:work")

    # P1 sends to P2 (causal chain: a -> d -> e -> g -> h, but f is spacelike)
    g, h = sys.send_message(1, 2, "g:send", "h:recv")

    # P0 does independent work (spacelike to g,h,f)
    i = sys.internal_event(0, "i:work")

    # P2 sends to P0 (closes the loop)
    j, k = sys.send_message(2, 0, "j:send", "k:recv")

    # --- Display ---
    print("\n--- All Events with Vector Clocks ---")
    for idx, ev in enumerate(sys.events):
        kind_str = f" [{ev.kind}]" if ev.kind != "internal" else ""
        print(f"  [{idx:2d}] P{ev.process} '{ev.label}' VC={ev.clock}{kind_str}")

    print("\n--- Spacetime Diagram ---")
    draw_spacetime(sys)

    # --- Full causal structure matrix ---
    print("\n--- Causal Structure (Pairwise Classification) ---")
    labels = [ev.label.split(":")[0] for ev in sys.events]
    print(f"{'':>10s}", end="")
    for lbl in labels:
        print(f"{lbl:>10s}", end="")
    print()
    for i_idx, ev_i in enumerate(sys.events):
        print(f"{labels[i_idx]:>10s}", end="")
        for j_idx, ev_j in enumerate(sys.events):
            if i_idx == j_idx:
                print(f"{'---':>10s}", end="")
            else:
                rel = classify(ev_i.clock, ev_j.clock)
                short = {"CAUSAL PAST": "past", "CAUSAL FUTURE": "future",
                         "SPACELIKE": "space"}[rel]
                print(f"{short:>10s}", end="")
        print()

    # --- Show light cones for selected events ---
    show_light_cone(sys, e)  # e:recv — mid-chain, has both past and spacelike
    show_light_cone(sys, f)  # f:work — independent work, lots of spacelike
    show_light_cone(sys, k)  # k:recv — late event, large past light cone

    print("\n" + "=" * 60)
    print("  INTERPRETATION")
    print("=" * 60)
    print("""
  The vector clock ordering defines exactly the same structure as
  light cones in special relativity:

  1. CAUSAL PAST: events whose information has reached this event
     (inside the past light cone). There is a chain of messages
     connecting them — like light signals connecting events in SR.

  2. CAUSAL FUTURE: events that will receive information from this
     event. This event's influence propagates forward through
     message chains — like a light cone expanding into the future.

  3. SPACELIKE: events with NO message chain connecting them in
     either direction. These events are CONCURRENT — they cannot
     influence each other, and there is no meaningful ordering.
     Different observers (processes) might "see" them in different
     orders, just as spacelike-separated events have no invariant
     temporal ordering in SR.

  The speed of light IS the message propagation speed. Both create
  a finite horizon for causal influence. Both reveal that some pairs
  of events simply HAVE no ordering — and that's not a bug, it's
  the fundamental structure of causality itself.
""")


if __name__ == "__main__":
    main()
