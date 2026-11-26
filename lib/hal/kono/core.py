#!/usr/bin/env python3
"""
KONO Core Classes for GRIP
Multi-dimensional memory substrate with evolutionary instance management.

Classes:
    - InstanceDNA: Skills, patterns, preferences fingerprint
    - GRIPInstance: Instance with fitness tracking and reproduction
    - Position4D: Spatial coordinates [x=project, y=topic, z=abstraction, t=time]
    - MemoryBlock: Memory unit with attestation-based permanence

Part of GRIP-KONO implementation (Gen 310).
"""

__version__ = "1.0.0"

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import hashlib
import json
import math


@dataclass
class InstanceDNA:
    """
    Fingerprint of a GRIP instance capturing its learned characteristics.

    Attributes:
        skills: List of skill names loaded in this instance
        patterns: Dict of pattern name -> frequency/weight
        preferences: Dict of preference key -> value
        project_context: Primary project this instance worked on
    """
    skills: List[str] = field(default_factory=list)
    patterns: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    project_context: str = ""

    def to_hash(self) -> str:
        """Generate deterministic hash of DNA for comparison."""
        data = {
            "skills": sorted(self.skills),
            "patterns": dict(sorted(self.patterns.items())),
            "preferences": dict(sorted(
                (k, str(v)) for k, v in self.preferences.items()
            )),
            "project": self.project_context
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def similarity(self, other: 'InstanceDNA') -> float:
        """
        Calculate similarity between two DNA instances.

        Uses Jaccard similarity for skills and weighted overlap for patterns.

        Returns:
            Float between 0 and 1 (1 = identical)
        """
        # Skill overlap (Jaccard)
        skills_self = set(self.skills)
        skills_other = set(other.skills)
        if skills_self or skills_other:
            skill_sim = len(skills_self & skills_other) / len(skills_self | skills_other)
        else:
            skill_sim = 1.0

        # Pattern overlap (weighted)
        patterns_self = set(self.patterns.keys())
        patterns_other = set(other.patterns.keys())
        if patterns_self or patterns_other:
            common = patterns_self & patterns_other
            if common:
                pattern_sim = sum(
                    min(self.patterns[p], other.patterns[p]) /
                    max(self.patterns[p], other.patterns[p])
                    for p in common
                ) / len(patterns_self | patterns_other)
            else:
                pattern_sim = 0.0
        else:
            pattern_sim = 1.0

        # Project context
        project_sim = 1.0 if self.project_context == other.project_context else 0.5

        # Weighted average
        return (skill_sim * 0.4) + (pattern_sim * 0.4) + (project_sim * 0.2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skills": self.skills,
            "patterns": self.patterns,
            "preferences": self.preferences,
            "project_context": self.project_context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceDNA':
        """Deserialize from dictionary."""
        return cls(
            skills=data.get("skills", []),
            patterns=data.get("patterns", {}),
            preferences=data.get("preferences", {}),
            project_context=data.get("project_context", "")
        )


@dataclass
class GRIPInstance:
    """
    A GRIP instance with DNA, fitness tracking, and reproduction capability.

    Fitness formula:
        fitness = base * (0.95 ^ days) * (1 + 0.1 * retrievals) * attestation_weight

    Attributes:
        id: Unique instance identifier (SHA prefix)
        generation: Generation number in lineage
        dna: Instance DNA fingerprint
        fitness: Current fitness score (0.0-1.0)
        parent_id: Parent instance ID (for lineage)
        connections: List of related instance IDs
        created_at: Creation timestamp
        last_active_at: Last activity timestamp
    """
    id: str
    generation: int
    dna: InstanceDNA = field(default_factory=InstanceDNA)
    fitness: float = 0.5
    parent_id: Optional[str] = None
    connections: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: Optional[datetime] = None

    def metabolize(self, days: float, access_boost: float = 0.0) -> None:
        """
        Apply time decay and access boost to fitness.

        Args:
            days: Days since last activity
            access_boost: Boost from retrievals (0.0-1.0)
        """
        # Time decay: shared constant with genome (subsystem_bridge.DECAY_RATE)
        try:
            from lib.subsystem_bridge import DECAY_RATE
        except ImportError:
            DECAY_RATE = 0.95
        time_decay = DECAY_RATE ** days

        # Access boost: 1 + 0.1 * retrievals (capped at 2.0)
        boost = 1 + min(access_boost, 1.0)

        self.fitness = min(1.0, max(0.0, self.fitness * time_decay * boost))

    def is_valid(self) -> bool:
        """Check if instance has sufficient fitness to be valid (>= 0.8)."""
        return self.fitness >= 0.8

    def reproduce(self, achievement: str = "") -> 'GRIPInstance':
        """
        Create child instance inheriting this instance's DNA.

        Args:
            achievement: Optional description of what triggered reproduction

        Returns:
            New GRIPInstance with incremented generation
        """
        # Generate child ID
        seed = f"{self.id}:{self.generation}:{datetime.now(timezone.utc).isoformat()}"
        child_id = hashlib.sha256(seed.encode()).hexdigest()[:8]

        # Inherit DNA with slight mutation (add achievement to patterns)
        child_dna = InstanceDNA(
            skills=self.dna.skills.copy(),
            patterns=self.dna.patterns.copy(),
            preferences=self.dna.preferences.copy(),
            project_context=self.dna.project_context
        )

        if achievement:
            pattern_key = f"achievement:{achievement[:50]}"
            child_dna.patterns[pattern_key] = child_dna.patterns.get(pattern_key, 0) + 1

        return GRIPInstance(
            id=child_id,
            generation=self.generation + 1,
            dna=child_dna,
            fitness=self.fitness * 0.9,  # Child starts slightly lower
            parent_id=self.id,
            connections=[self.id],
            created_at=datetime.now(timezone.utc)
        )

    def similarity(self, other: 'GRIPInstance') -> float:
        """Calculate similarity to another instance via DNA comparison."""
        return self.dna.similarity(other.dna)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "generation": self.generation,
            "dna": self.dna.to_dict(),
            "fitness": self.fitness,
            "parent_id": self.parent_id,
            "connections": self.connections,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GRIPInstance':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            generation=data["generation"],
            dna=InstanceDNA.from_dict(data.get("dna", {})),
            fitness=data.get("fitness", 0.5),
            parent_id=data.get("parent_id"),
            connections=data.get("connections", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            last_active_at=datetime.fromisoformat(data["last_active_at"]) if data.get("last_active_at") else None
        )


@dataclass
class Position4D:
    """
    4D spatial coordinates for memory placement.

    Dimensions:
        x: Project/domain (0-N based on kono_domains)
        y: Topic/concept (0-N based on kono_topics)
        z: Abstraction level (0=concrete, 9=abstract)
        t: Temporal (epoch timestamp)
    """
    x: float = 0.0  # project/domain
    y: float = 0.0  # topic/concept
    z: float = 5.0  # abstraction level (default mid)
    t: float = 0.0  # timestamp

    def distance(self, other: 'Position4D', weights: Tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.1)) -> float:
        """
        Calculate weighted Euclidean distance to another position.

        Args:
            other: Target position
            weights: (wx, wy, wz, wt) weights for each dimension

        Returns:
            Weighted distance (lower = closer)
        """
        wx, wy, wz, wt = weights
        return math.sqrt(
            wx * (self.x - other.x) ** 2 +
            wy * (self.y - other.y) ** 2 +
            wz * (self.z - other.z) ** 2 +
            wt * (self.t - other.t) ** 2
        )

    @classmethod
    def from_embedding(
        cls,
        embedding: List[float],
        timestamp: float,
        project_x: int = 0,
        topic_y: int = 0
    ) -> 'Position4D':
        """
        Create position from embedding vector.

        Uses PCA-like projection to derive z (abstraction) from embedding.

        Args:
            embedding: Vector embedding
            timestamp: Epoch timestamp
            project_x: Project x-coordinate (from kono_domains)
            topic_y: Topic y-coordinate (from kono_topics)

        Returns:
            Position4D with calculated coordinates
        """
        # Derive abstraction level from embedding variance
        # Higher variance = more abstract concept
        if embedding:
            mean = sum(embedding) / len(embedding)
            variance = sum((e - mean) ** 2 for e in embedding) / len(embedding)
            # Map variance to 0-9 scale (calibrated for typical embedding ranges)
            z = min(9.0, max(0.0, variance * 100))
        else:
            z = 5.0

        return cls(
            x=float(project_x),
            y=float(topic_y),
            z=z,
            t=timestamp
        )

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as tuple for R-tree operations."""
        return (self.x, self.y, self.z, self.t)


@dataclass
class MemoryBlock:
    """
    A block of memory with attestation-based permanence.

    Permanence is achieved when attestation_count >= 3 (configurable).

    Attributes:
        id: Unique block identifier
        position: 4D spatial position
        chunks: List of embedding IDs in this block
        attestations: List of (instance_id, fitness_at_attestation) tuples
        dimension: Memory dimension (0-11)
        prev_hash: Hash of previous block (for chain integrity)
        hash: This block's hash
        fitness_proof: Combined fitness proof from attestations
        is_permanent: Whether block has achieved permanence
    """
    id: str
    position: Position4D
    chunks: List[int] = field(default_factory=list)
    attestations: List[Tuple[str, float]] = field(default_factory=list)
    dimension: int = 0
    prev_hash: str = ""
    hash: str = ""
    fitness_proof: float = 0.0
    is_permanent: bool = False

    def __post_init__(self):
        """Calculate hash if not provided."""
        if not self.hash:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Generate deterministic hash of block content."""
        data = {
            "id": self.id,
            "position": self.position.to_tuple(),
            "chunks": sorted(self.chunks),
            "dimension": self.dimension,
            "prev_hash": self.prev_hash
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_valid(self) -> bool:
        """Check if block has sufficient fitness proof (>= 0.8)."""
        return self.fitness_proof >= 0.8

    def add_attestation(self, instance_id: str, fitness: float) -> None:
        """
        Add attestation from an instance.

        Updates fitness_proof and checks permanence threshold.

        Args:
            instance_id: Attesting instance ID
            fitness: Instance's fitness at time of attestation
        """
        # Avoid duplicate attestations
        existing_ids = {a[0] for a in self.attestations}
        if instance_id in existing_ids:
            return

        self.attestations.append((instance_id, fitness))

        # Update fitness proof (weighted average of attesting instance fitnesses)
        total_fitness = sum(f for _, f in self.attestations)
        self.fitness_proof = total_fitness / len(self.attestations)

        # Check permanence (3+ attestations with sufficient fitness)
        if len(self.attestations) >= 3 and self.fitness_proof >= 0.5:
            self.is_permanent = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
                "t": self.position.t
            },
            "chunks": self.chunks,
            "attestations": self.attestations,
            "dimension": self.dimension,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "fitness_proof": self.fitness_proof,
            "is_permanent": self.is_permanent
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Deserialize from dictionary."""
        pos_data = data.get("position", {})
        return cls(
            id=data["id"],
            position=Position4D(
                x=pos_data.get("x", 0),
                y=pos_data.get("y", 0),
                z=pos_data.get("z", 5),
                t=pos_data.get("t", 0)
            ),
            chunks=data.get("chunks", []),
            attestations=[tuple(a) for a in data.get("attestations", [])],
            dimension=data.get("dimension", 0),
            prev_hash=data.get("prev_hash", ""),
            hash=data.get("hash", ""),
            fitness_proof=data.get("fitness_proof", 0.0),
            is_permanent=data.get("is_permanent", False)
        )


def calculate_fitness(
    base: float,
    days_since_access: float,
    retrievals: int,
    attestation_weight: float = 1.0
) -> float:
    """
    Calculate fitness using KONO formula.

    Formula: fitness = base * (0.95^days) * (1 + 0.1*retrievals) * attestation

    Args:
        base: Base fitness score (0.0-1.0)
        days_since_access: Days since last access
        retrievals: Number of times retrieved
        attestation_weight: Weight from attestations (0.0-1.0)

    Returns:
        Calculated fitness (capped at 1.0)
    """
    time_decay = 0.95 ** days_since_access
    access_boost = 1 + (0.1 * min(retrievals, 10))  # Cap at 2x
    return min(1.0, base * time_decay * access_boost * attestation_weight)


# Dimension routing map: memory_type -> (dim_start, dim_mid, dim_end)
# Factual 0-2: concrete facts, prompts, definitions
# Procedural 3-5: how-to knowledge, plans, skills, workflows
# Episodic 6-8: session events, errors, checkpoints, focus cards
# Emergent 9-11: cross-session patterns, learnings, insights
DIMENSION_ROUTES = {
    'factual': (0, 1, 2),
    'procedural': (3, 4, 5),
    'episodic': (6, 7, 8),
    'emergent': (9, 10, 11)
}

# Content type to specific dimension. Unrecognized types default to 0.
# See docs/KONO-USAGE-GUIDE.md for full schema documentation.
TYPE_TO_DIMENSION = {
    'prompt': 0,       # Factual: user queries
    'response': 3,     # Procedural: agent outputs
    'concept': 6,      # Episodic: abstract concepts
    'checkpoint': 6,   # Episodic: session state snapshots
    'plan': 3,         # Procedural: implementation plans
    'skill': 3,        # Procedural: skill protocols
    'error': 6,        # Episodic: error patterns
    'learning': 9,     # Emergent: insights from detection
    'pattern': 9,      # Emergent: recurring cross-session patterns
    'focus_card': 7,   # Episodic: active session work cards
}


def route_dimension(content_type: str) -> int:
    """
    Route content type to appropriate dimension.

    Args:
        content_type: Type of content being stored

    Returns:
        Dimension number (0-11)
    """
    return TYPE_TO_DIMENSION.get(content_type, 0)


def main():
    """CLI for KONO classes testing."""
    import argparse

    parser = argparse.ArgumentParser(description="KONO Core Classes")
    subparsers = parser.add_subparsers(dest="command")

    # Test DNA
    dna_parser = subparsers.add_parser("test-dna", help="Test InstanceDNA")
    dna_parser.add_argument("--skills", nargs="+", default=["ultrathink", "asking-users"])

    # Test Instance
    inst_parser = subparsers.add_parser("test-instance", help="Test GRIPInstance")
    inst_parser.add_argument("--id", default="test1234")
    inst_parser.add_argument("--gen", type=int, default=1)

    # Test Position
    pos_parser = subparsers.add_parser("test-position", help="Test Position4D")

    # Test Block
    block_parser = subparsers.add_parser("test-block", help="Test MemoryBlock")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "test-dna":
        dna1 = InstanceDNA(skills=args.skills, patterns={"test": 1.0})
        dna2 = InstanceDNA(skills=args.skills[:1], patterns={"test": 0.5})
        print(f"DNA1 hash: {dna1.to_hash()}")
        print(f"DNA2 hash: {dna2.to_hash()}")
        print(f"Similarity: {dna1.similarity(dna2):.4f}")

    elif args.command == "test-instance":
        inst = GRIPInstance(id=args.id, generation=args.gen)
        print(f"Instance: {inst.id}, Gen {inst.generation}")
        print(f"Fitness: {inst.fitness}")
        print(f"Valid: {inst.is_valid()}")

        # Test reproduction
        child = inst.reproduce("test achievement")
        print(f"Child: {child.id}, Gen {child.generation}")
        print(f"Similarity: {inst.similarity(child):.4f}")

    elif args.command == "test-position":
        pos1 = Position4D(x=1, y=2, z=5, t=1000)
        pos2 = Position4D(x=2, y=3, z=6, t=1100)
        print(f"Pos1: {pos1.to_tuple()}")
        print(f"Pos2: {pos2.to_tuple()}")
        print(f"Distance: {pos1.distance(pos2):.4f}")

    elif args.command == "test-block":
        pos = Position4D(x=0, y=0, z=5, t=1000)
        block = MemoryBlock(id="block1", position=pos, chunks=[1, 2, 3])
        print(f"Block: {block.id}")
        print(f"Hash: {block.hash}")

        # Test attestation
        block.add_attestation("inst1", 0.9)
        block.add_attestation("inst2", 0.85)
        block.add_attestation("inst3", 0.8)
        print(f"Attestations: {len(block.attestations)}")
        print(f"Fitness proof: {block.fitness_proof:.4f}")
        print(f"Permanent: {block.is_permanent}")


if __name__ == "__main__":
    main()
