"""Fact dataset utilities for knowledge representation experiments.

This module provides tools for creating, loading, and managing datasets of
factual statements for probing model knowledge representation.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """Represents a factual triple.

    Attributes:
        subject: The subject entity (e.g., "Eiffel Tower")
        relation: The relation type (e.g., "located_in", "invented_by")
        object: The object entity (e.g., "Paris")
        is_true: Whether this fact is true or counterfactual
        metadata: Optional additional information
    """

    subject: str
    relation: str
    object: str
    is_true: bool = True
    metadata: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        """Create Fact from dictionary."""
        return cls(**data)

    def to_prompt(self, template: str | None = None) -> str:
        """Convert fact to a natural language prompt.

        Args:
            template: Template string with {subject}, {relation}, {object} placeholders.
                     If None, uses a default template based on relation type.

        Returns:
            Natural language prompt string.
        """
        if template is not None:
            return template.format(subject=self.subject, relation=self.relation, object=self.object)

        # Default templates based on common relation types
        relation_templates = {
            "located_in": "{subject} is located in {object}",
            "capital_of": "{subject} is the capital of {object}",
            "invented_by": "{subject} was invented by {object}",
            "born_in": "{subject} was born in {object}",
            "nationality": "{subject} has nationality {object}",
            "speaks": "{subject} speaks {object}",
            "profession": "{subject} is a {object}",
            "works_for": "{subject} works for {object}",
            "parent_company": "{subject} is owned by {object}",
            "ceo_of": "{subject} is the CEO of {object}",
        }

        template = relation_templates.get(
            self.relation, "{subject} {relation} {object}"  # Fallback template
        )

        return template.format(
            subject=self.subject, relation=self.relation.replace("_", " "), object=self.object
        )

    def negate(self, new_object: str) -> "Fact":
        """Create a counterfactual version of this fact.

        Args:
            new_object: The incorrect object to use.

        Returns:
            New Fact instance with is_true=False and updated object.
        """
        return Fact(
            subject=self.subject,
            relation=self.relation,
            object=new_object,
            is_true=False,
            metadata={
                "original_object": self.object,
                "type": "counterfactual",
                **(self.metadata or {}),
            },
        )


class FactDataset:
    """Dataset of factual statements for knowledge probing.

    Example:
        >>> dataset = FactDataset()
        >>> dataset.add_fact("Paris", "capital_of", "France")
        >>> dataset.add_fact("Berlin", "capital_of", "Germany")
        >>> prompts = dataset.to_prompts()
        >>> print(prompts[0])  # "Paris is the capital of France"
    """

    def __init__(self, facts: list[Fact] | None = None):
        """Initialize fact dataset.

        Args:
            facts: Optional list of Fact objects to initialize with.
        """
        self.facts: list[Fact] = facts or []
        logger.info(f"Initialized FactDataset with {len(self.facts)} facts")

    def add_fact(
        self,
        subject: str,
        relation: str,
        object: str,
        is_true: bool = True,
        **metadata,
    ) -> None:
        """Add a fact to the dataset.

        Args:
            subject: Subject entity
            relation: Relation type
            object: Object entity
            is_true: Whether the fact is true
            **metadata: Additional metadata to store
        """
        fact = Fact(
            subject=subject,
            relation=relation,
            object=object,
            is_true=is_true,
            metadata=metadata if metadata else None,
        )
        self.facts.append(fact)

    def add_fact_with_counterfactual(
        self,
        subject: str,
        relation: str,
        true_object: str,
        false_object: str,
    ) -> tuple[Fact, Fact]:
        """Add both a true fact and its counterfactual.

        Args:
            subject: Subject entity
            relation: Relation type
            true_object: The correct object
            false_object: An incorrect object

        Returns:
            Tuple of (true_fact, false_fact)
        """
        true_fact = Fact(subject, relation, true_object, is_true=True)
        false_fact = true_fact.negate(false_object)

        self.facts.extend([true_fact, false_fact])
        return true_fact, false_fact

    def filter(
        self,
        relation: str | None = None,
        is_true: bool | None = None,
    ) -> "FactDataset":
        """Filter facts based on criteria.

        Args:
            relation: Filter by relation type
            is_true: Filter by truthfulness

        Returns:
            New FactDataset with filtered facts
        """
        filtered_facts = self.facts

        if relation is not None:
            filtered_facts = [f for f in filtered_facts if f.relation == relation]

        if is_true is not None:
            filtered_facts = [f for f in filtered_facts if f.is_true == is_true]

        return FactDataset(filtered_facts)

    def to_prompts(
        self,
        template: str | None = None,
        include_labels: bool = False,
    ) -> list[str] | list[tuple[str, bool]]:
        """Convert all facts to prompts.

        Args:
            template: Optional template string for all facts
            include_labels: If True, return (prompt, is_true) tuples

        Returns:
            List of prompts or (prompt, label) tuples
        """
        if include_labels:
            return [(f.to_prompt(template), f.is_true) for f in self.facts]
        else:
            return [f.to_prompt(template) for f in self.facts]

    def get_prompt_pairs(self) -> list[tuple[str, str]]:
        """Get pairs of (true_prompt, false_prompt) for the same subject-relation.

        Returns:
            List of (true_prompt, false_prompt) tuples

        Raises:
            ValueError: If facts cannot be paired (missing true or false version)
        """
        # Group facts by (subject, relation)
        fact_groups: dict[tuple[str, str], dict[bool, Fact]] = {}

        for fact in self.facts:
            key = (fact.subject, fact.relation)
            if key not in fact_groups:
                fact_groups[key] = {}
            fact_groups[key][fact.is_true] = fact

        # Create pairs
        pairs = []
        for key, facts in fact_groups.items():
            if True not in facts or False not in facts:
                logger.warning(f"Skipping {key}: missing true or false version")
                continue

            true_prompt = facts[True].to_prompt()
            false_prompt = facts[False].to_prompt()
            pairs.append((true_prompt, false_prompt))

        logger.info(f"Created {len(pairs)} prompt pairs")
        return pairs

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file.

        Args:
            path: Path to save the JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [fact.to_dict() for fact in self.facts]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.facts)} facts to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FactDataset":
        """Load dataset from JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            FactDataset instance
        """
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        facts = [Fact.from_dict(item) for item in data]
        logger.info(f"Loaded {len(facts)} facts from {path}")

        return cls(facts)

    def __len__(self) -> int:
        """Return number of facts."""
        return len(self.facts)

    def __getitem__(self, idx: int) -> Fact:
        """Get fact by index."""
        return self.facts[idx]

    def __repr__(self) -> str:
        """String representation."""
        n_true = sum(1 for f in self.facts if f.is_true)
        n_false = len(self.facts) - n_true
        return f"FactDataset(total={len(self)}, true={n_true}, false={n_false})"


def create_sample_dataset() -> FactDataset:
    """Create a sample fact dataset for demonstration.

    Returns:
        FactDataset with example facts about geography, people, and companies.
    """
    dataset = FactDataset()

    # Geographic facts
    dataset.add_fact_with_counterfactual("Eiffel Tower", "located_in", "Paris", "London")
    dataset.add_fact_with_counterfactual("Statue of Liberty", "located_in", "New York", "Boston")
    dataset.add_fact_with_counterfactual("Big Ben", "located_in", "London", "Paris")

    # Capitals
    dataset.add_fact_with_counterfactual("Paris", "capital_of", "France", "Germany")
    dataset.add_fact_with_counterfactual("Berlin", "capital_of", "Germany", "France")
    dataset.add_fact_with_counterfactual("Tokyo", "capital_of", "Japan", "China")

    # Inventions
    dataset.add_fact_with_counterfactual(
        "telephone", "invented_by", "Alexander Graham Bell", "Thomas Edison"
    )
    dataset.add_fact_with_counterfactual(
        "light bulb", "invented_by", "Thomas Edison", "Alexander Graham Bell"
    )

    # People
    dataset.add_fact_with_counterfactual("Albert Einstein", "born_in", "Germany", "Switzerland")
    dataset.add_fact_with_counterfactual("Marie Curie", "born_in", "Poland", "France")

    # Companies
    dataset.add_fact_with_counterfactual("Tim Cook", "ceo_of", "Apple", "Microsoft")
    dataset.add_fact_with_counterfactual("Satya Nadella", "ceo_of", "Microsoft", "Apple")

    logger.info(f"Created sample dataset: {dataset}")
    return dataset


def load_or_create_dataset(
    path: str | Path,
    create_if_missing: bool = True,
) -> FactDataset:
    """Load dataset from file, or create sample if it doesn't exist.

    Args:
        path: Path to dataset JSON file
        create_if_missing: If True, create and save sample dataset if file not found

    Returns:
        FactDataset instance
    """
    path = Path(path)

    if path.exists():
        return FactDataset.load(path)
    elif create_if_missing:
        logger.info(f"Dataset not found at {path}, creating sample dataset")
        dataset = create_sample_dataset()
        dataset.save(path)
        return dataset
    else:
        raise FileNotFoundError(f"Dataset not found at {path}")
