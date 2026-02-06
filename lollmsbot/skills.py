"""
Skills Module - LollmsBot's Learned Capability System

Skills are reusable, composable workflows that encode "how to do things" - from
simple tasks like "send a formatted email" to complex orchestrations like
"research, write, and publish a blog post."

Key characteristics:
- Self-documenting: Each skill explains what it does, when to use it, what it needs
- Dependency-aware: Skills declare what other skills/tools they require
- Versioned: Skills evolve, with history and rollback capability
- Composable: Skills can call other skills, building complex workflows
- Learnable: LollmsBot can create new skills from demonstration or description
- Auditable: All skill executions are logged for review

Architecture:
- Skill: Atomic unit of capability
- SkillRegistry: Manages skill discovery, loading, versioning
- SkillExecutor: Runs skills with proper dependency injection and logging
- SkillLearner: Creates new skills from examples or specifications
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import logging
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Type, Union

from lollmsbot.agent import Agent, Tool, ToolResult
from lollmsbot.guardian import get_guardian, GuardianAction


logger = logging.getLogger("lollmsbot.skills")


class SkillComplexity(Enum):
    """Complexity classification for skills."""
    TRIVIAL = auto()    # Single operation, no decisions (e.g., "format date")
    SIMPLE = auto()     # Linear sequence, no branching (e.g., "send email")
    MODERATE = auto()   # Conditional logic, error handling (e.g., "process invoice")
    COMPLEX = auto()    # Multi-step orchestration, state management (e.g., "onboard employee")
    SYSTEM = auto()     # Meta-skills that create/modify other skills


@dataclass
class SkillParameter:
    """Definition of a skill's input parameter."""
    name: str
    type: str  # JSON Schema type: string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    examples: List[Any] = field(default_factory=list)
    validation_regex: Optional[str] = None  # Pattern for string validation
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema property."""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.examples:
            schema["examples"] = self.examples
        if self.validation_regex:
            schema["pattern"] = self.validation_regex
        return schema


@dataclass
class SkillOutput:
    """Definition of a skill's output."""
    name: str
    type: str
    description: str
    always_present: bool = True
    
    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
        }


@dataclass
class SkillDependency:
    """Declaration of what this skill needs to function."""
    kind: str  # "tool", "skill", "api", "file", "env"
    name: str
    version_constraint: Optional[str] = None  # e.g., ">=1.0.0", "filesystem>=2.0"
    optional: bool = False
    reason: str = ""  # Why is this needed?
    
    def is_satisfied(self, available_tools: Set[str], available_skills: Set[str]) -> bool:
        """Check if this dependency is currently available."""
        if self.kind == "tool":
            return self.name in available_tools
        elif self.kind == "skill":
            return self.name in available_skills
        # Other kinds require runtime checks
        return True


@dataclass
class SkillExample:
    """Demonstration of how to use the skill."""
    description: str
    input_params: Dict[str, Any]
    expected_output: Dict[str, Any]
    notes: str = ""  # Special considerations, edge cases


@dataclass
class SkillMetadata:
    """Comprehensive metadata about a skill."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    long_description: str = ""
    author: str = "lollmsbot"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    complexity: SkillComplexity = SkillComplexity.SIMPLE
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Documentation
    when_to_use: str = ""  # Guidance on appropriate usage
    when_not_to_use: str = ""  # Anti-patterns, alternatives
    prerequisites: List[str] = field(default_factory=list)  # Human knowledge needed
    estimated_duration: Optional[str] = None  # "30 seconds", "2-5 minutes"
    error_handling: str = "basic"  # "minimal", "basic", "robust", "comprehensive"
    
    # Technical
    parameters: List[SkillParameter] = field(default_factory=list)
    outputs: List[SkillOutput] = field(default_factory=list)
    dependencies: List[SkillDependency] = field(default_factory=list)
    examples: List[SkillExample] = field(default_factory=list)
    
    # Provenance
    parent_skill: Optional[str] = None  # If this was derived from another
    learning_method: Optional[str] = None  # "demonstration", "description", "abstraction", "composition"
    confidence_score: float = 1.0  # 0-1, how sure we are this works
    
    # Runtime stats
    execution_count: int = 0
    success_rate: float = 0.0
    last_executed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        # Convert enums
        data["complexity"] = self.complexity.name
        data["created_at"] = self.created_at.isoformat()
        data["modified_at"] = self.modified_at.isoformat()
        data["last_executed"] = self.last_executed.isoformat() if self.last_executed else None
        # Convert nested dataclasses
        data["parameters"] = [asdict(p) for p in self.parameters]
        data["outputs"] = [asdict(o) for o in self.outputs]
        data["dependencies"] = [asdict(d) for d in self.dependencies]
        data["examples"] = [asdict(e) for e in self.examples]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """Deserialize from dictionary."""
        # Reconstruct enums
        complexity = SkillComplexity[data.get("complexity", "SIMPLE")]
        # Reconstruct datetimes
        created_at = datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else datetime.now()
        modified_at = datetime.fromisoformat(data.get("modified_at")) if data.get("modified_at") else datetime.now()
        last_executed = datetime.fromisoformat(data["last_executed"]) if data.get("last_executed") else None
        # Reconstruct nested classes
        parameters = [SkillParameter(**p) for p in data.get("parameters", [])]
        outputs = [SkillOutput(**o) for o in data.get("outputs", [])]
        dependencies = [SkillDependency(**d) for d in data.get("dependencies", [])]
        examples = [SkillExample(**e) for e in data.get("examples", [])]
        
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            long_description=data.get("long_description", ""),
            author=data.get("author", "lollmsbot"),
            created_at=created_at,
            modified_at=modified_at,
            complexity=complexity,
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            when_to_use=data.get("when_to_use", ""),
            when_not_to_use=data.get("when_not_to_use", ""),
            prerequisites=data.get("prerequisites", []),
            estimated_duration=data.get("estimated_duration"),
            error_handling=data.get("error_handling", "basic"),
            parameters=parameters,
            outputs=outputs,
            dependencies=dependencies,
            examples=examples,
            parent_skill=data.get("parent_skill"),
            learning_method=data.get("learning_method"),
            confidence_score=data.get("confidence_score", 1.0),
            execution_count=data.get("execution_count", 0),
            success_rate=data.get("success_rate", 0.0),
            last_executed=last_executed,
        )


@dataclass
class SkillExecutionRecord:
    """Log entry for a single skill execution."""
    execution_id: str
    skill_name: str
    skill_version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    input_params: Dict[str, Any] = field(default_factory=dict, repr=False)  # Sensitive data
    output_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    steps_executed: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    skills_called: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    guardian_events: List[str] = field(default_factory=list)  # Security review flags
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "error_message": self.error_message,
            "steps_executed": self.steps_executed,
            "tools_used": self.tools_used,
            "skills_called": self.skills_called,
            "duration_seconds": self.duration_seconds,
            "guardian_events": self.guardian_events,
        }


class Skill:
    """
    A reusable, documented, versioned capability.
    
    Skills can be:
    - Code-based: Python functions with decorators
    - Template-based: Jinja2 templates with parameter substitution
    - LLM-based: Natural language descriptions executed by LLM reasoning
    - Composite: Orchestrations of other skills
    """
    
    def __init__(
        self,
        metadata: SkillMetadata,
        implementation: Union[Callable, str, Dict[str, Any]],
        implementation_type: str = "code",  # "code", "template", "llm", "composite"
    ):
        self.metadata = metadata
        self.implementation = implementation
        self.implementation_type = implementation_type
        
        # Runtime state
        self._compiled_code: Optional[Any] = None
        self._template: Optional[Any] = None
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate provided inputs against parameter schema."""
        errors = []
        
        for param in self.metadata.parameters:
            if param.name not in inputs:
                if param.required:
                    errors.append(f"Missing required parameter: {param.name}")
                continue
            
            value = inputs[param.name]
            
            # Type checking
            if param.type == "string" and not isinstance(value, str):
                errors.append(f"{param.name}: expected string, got {type(value).__name__}")
            elif param.type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{param.name}: expected number, got {type(value).__name__}")
            elif param.type == "boolean" and not isinstance(value, bool):
                errors.append(f"{param.name}: expected boolean, got {type(value).__name__}")
            elif param.type == "array" and not isinstance(value, list):
                errors.append(f"{param.name}: expected array, got {type(value).__name__}")
            elif param.type == "object" and not isinstance(value, dict):
                errors.append(f"{param.name}: expected object, got {type(value).__name__}")
            
            # Pattern validation
            if param.validation_regex and isinstance(value, str):
                if not re.match(param.validation_regex, value):
                    errors.append(f"{param.name}: does not match required pattern")
        
        # Check for unknown parameters
        known = {p.name for p in self.metadata.parameters}
        unknown = set(inputs.keys()) - known
        if unknown:
            errors.append(f"Unknown parameters: {', '.join(unknown)}")
        
        return len(errors) == 0, errors
    
    def check_dependencies(self, available_tools: Set[str], available_skills: Set[str]) -> Tuple[bool, List[str]]:
        """Check if all dependencies are satisfied."""
        missing = []
        for dep in self.metadata.dependencies:
            if not dep.optional and not dep.is_satisfied(available_tools, available_skills):
                missing.append(f"{dep.kind}:{dep.name} ({dep.reason})")
        return len(missing) == 0, missing
    
    def to_prompt_description(self) -> str:
        """Generate natural language description for LLM tool selection."""
        lines = [
            f"## Skill: {self.name}",
            f"**Description:** {self.metadata.description}",
            "",
            f"**When to use:** {self.metadata.when_to_use or 'Appropriate when ' + self.metadata.description.lower()}",
        ]
        
        if self.metadata.when_not_to_use:
            lines.append(f"**When NOT to use:** {self.metadata.when_not_to_use}")
        
        lines.extend([
            "",
            "**Parameters:**",
        ])
        for param in self.metadata.parameters:
            req = " (required)" if param.required else " (optional)"
            default = f", default: {param.default}" if param.default is not None else ""
            lines.append(f"- `{param.name}` ({param.type}){req}: {param.description}{default}")
        
        if self.metadata.examples:
            lines.extend(["", "**Example:**"])
            ex = self.metadata.examples[0]
            lines.append(f"Input: {json.dumps(ex.input_params, indent=2)}")
            lines.append(f"Output: {json.dumps(ex.expected_output, indent=2)}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete skill."""
        return {
            "metadata": self.metadata.to_dict(),
            "implementation_type": self.implementation_type,
            "implementation": self._serialize_implementation(),
        }
    
    def _serialize_implementation(self) -> Any:
        """Serialize implementation based on type."""
        if self.implementation_type == "code" and callable(self.implementation):
            # Store source code
            try:
                return inspect.getsource(self.implementation)
            except (OSError, TypeError):
                return "# Source not available"
        return self.implementation


class SkillRegistry:
    """
    Central registry for all skills.
    
    Manages skill discovery, loading, versioning, and dependency resolution.
    Provides skill search and recommendation capabilities.
    """
    
    DEFAULT_SKILLS_DIR = Path.home() / ".lollmsbot" / "skills"
    
    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or self.DEFAULT_SKILLS_DIR
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self._skills: Dict[str, Skill] = {}  # name -> latest version
        self._version_history: Dict[str, List[Skill]] = {}  # name -> all versions
        self._categories: Dict[str, Set[str]] = {}  # category -> skill names
        self._tags: Dict[str, Set[str]] = {}  # tag -> skill names
        
        # Index for search
        self._search_index: Dict[str, Set[str]] = {}  # word -> skill names
        
        # Load built-in and user skills
        self._load_built_in_skills()
        self._load_user_skills()
    
    def _load_built_in_skills(self) -> None:
        """Register essential built-in skills."""
        built_ins = [
            self._create_file_organizer_skill(),
            self._create_research_synthesizer_skill(),
            self._create_meeting_prep_skill(),
            self._create_code_review_skill(),
            self._create_learning_skill(),  # Meta-skill for learning new skills
        ]
        for skill in built_ins:
            self.register(skill, is_builtin=True)
    
    def _create_file_organizer_skill(self) -> Skill:
        """Create the file organizer skill."""
        metadata = SkillMetadata(
            name="organize_files",
            version="1.0.0",
            description="Organize files in a directory by date, type, or custom rules",
            long_description="""
            Automatically organizes files in a specified directory using intelligent categorization.
            Can organize by: date (YYYY-MM folders), type (extension-based categories), 
            or custom rules (pattern matching).
            """,
            complexity=SkillComplexity.SIMPLE,
            tags=["files", "organization", "automation"],
            categories=["productivity", "file-management"],
            when_to_use="When downloads folder is cluttered, before archiving projects, when setting up new workspace",
            when_not_to_use="For system directories, when files are actively being used by running processes",
            estimated_duration="1-5 minutes depending on file count",
            parameters=[
                SkillParameter("source_dir", "string", "Directory containing files to organize", required=True),
                SkillParameter("method", "string", "Organization method: 'date', 'type', or 'custom'", 
                              required=True, examples=["date", "type"]),
                SkillParameter("custom_rules", "object", "For 'custom' method: {pattern: destination_folder}", 
                              required=False, default={}),
                SkillParameter("dry_run", "boolean", "Preview changes without moving files", 
                              required=False, default=True),
            ],
            outputs=[
                SkillOutput("moved_files", "array", "List of {source, destination} for moved files"),
                SkillOutput("stats", "object", "Summary: {total_files, organized_by_category}"),
            ],
            dependencies=[
                SkillDependency("tool", "filesystem", reason="Needs to move and organize files"),
            ],
            examples=[
                SkillExample(
                    description="Organize downloads by file type",
                    input_params={
                        "source_dir": "~/Downloads",
                        "method": "type",
                        "dry_run": False,
                    },
                    expected_output={
                        "moved_files": [{"source": "~/Downloads/report.pdf", "destination": "~/Downloads/PDFs/report.pdf"}],
                        "stats": {"total_files": 42, "organized_by_category": {"PDFs": 5, "Images": 12, "Archives": 3}},
                    },
                ),
            ],
        )
        
        # Implementation is a JSON description for LLM-guided execution
        implementation = {
            "execution_plan": [
                {"step": "analyze", "description": "List all files in source_dir and categorize by method"},
                {"step": "preview", "description": "If dry_run, show planned organization without moving"},
                {"step": "organize", "description": "Create category folders and move files"},
                {"step": "verify", "description": "Confirm all moves successful, report stats"},
            ],
            "error_handling": {
                "permission_denied": "Skip file and continue, report at end",
                "filename_collision": "Append number to create unique name",
                "disk_full": "Stop immediately, report partial completion",
            },
        }
        
        return Skill(metadata, implementation, "composite")
    
    def _create_research_synthesizer_skill(self) -> Skill:
        """Create the research synthesis skill."""
        metadata = SkillMetadata(
            name="synthesize_research",
            version="1.0.0",
            description="Research a topic across multiple sources and synthesize findings",
            long_description="""
            Performs comprehensive research by querying multiple sources (web search, 
            knowledge base, documents), then synthesizes findings into structured output
            with citations, confidence levels, and gaps identified.
            """,
            complexity=SkillComplexity.COMPLEX,
            tags=["research", "synthesis", "knowledge", "learning"],
            categories=["research", "knowledge-work"],
            when_to_use="When exploring new topic, preparing reports, validating claims, learning efficiently",
            when_not_to_use="For time-sensitive decisions, when primary sources are required, for legal/medical advice",
            estimated_duration="2-10 minutes depending on breadth",
            parameters=[
                SkillParameter("topic", "string", "Research topic or question", required=True),
                SkillParameter("depth", "string", "Research depth: 'quick', 'standard', 'comprehensive'", 
                              required=False, default="standard"),
                SkillParameter("sources", "array", "Preferred sources: 'web', 'documents', 'kb'", 
                              required=False, default=["web", "kb"]),
                SkillParameter("output_format", "string", "Output structure: 'summary', 'report', 'outline', 'qa'", 
                              required=False, default="summary"),
            ],
            outputs=[
                SkillOutput("synthesis", "object", "Structured findings with main points, evidence, confidence"),
                SkillOutput("sources_used", "array", "List of sources consulted with relevance scores"),
                SkillOutput("gaps", "array", "Questions that couldn't be answered with available sources"),
                SkillOutput("follow_up", "array", "Suggested next research directions"),
            ],
            dependencies=[
                SkillDependency("tool", "http", reason="Web search and API queries"),
                SkillDependency("tool", "filesystem", reason="Reading local documents"),
                SkillDependency("skill", "evaluate_source_credibility", optional=True, 
                               reason="Better source quality assessment"),
            ],
            examples=[
                SkillExample(
                    description="Quick research on Python async patterns",
                    input_params={
                        "topic": "Python asyncio best practices for web services",
                        "depth": "quick",
                        "output_format": "summary",
                    },
                    expected_output={
                        "synthesis": {"main_points": ["Use asyncio.create_task for fire-and-forget", ...]},
                        "sources_used": [{"url": "...", "relevance": 0.95}],
                        "gaps": ["Performance comparison with threading"],
                        "follow_up": ["Research asyncio vs trio frameworks"],
                    },
                ),
            ],
        )
        
        implementation = {
            "execution_plan": [
                {"step": "decompose", "description": "Break topic into sub-questions and search queries"},
                {"step": "search", "description": "Query each source type with appropriate queries"},
                {"step": "evaluate", "description": "Assess source credibility and extract key claims"},
                {"step": "synthesize", "description": "Integrate findings, resolve conflicts, assign confidence"},
                {"step": "structure", "description": "Format according to output_format with citations"},
            ],
        }
        
        return Skill(metadata, implementation, "llm")
    
    def _create_meeting_prep_skill(self) -> Skill:
        """Create the meeting preparation skill."""
        metadata = SkillMetadata(
            name="prepare_meeting",
            version="1.0.0",
            description="Prepare comprehensive briefing for upcoming meeting",
            complexity=SkillComplexity.MODERATE,
            tags=["meetings", "productivity", "preparation"],
            categories=["productivity", "communication"],
            when_to_use="Before important meetings, when joining new project, when meeting unfamiliar attendees",
            parameters=[
                SkillParameter("meeting_title", "string", "Title or topic of meeting", required=True),
                SkillParameter("attendees", "array", "List of attendee names/roles", required=False, default=[]),
                SkillParameter("my_role", "string", "Your role/perspective in this meeting", required=False, default="participant"),
                SkillParameter("duration_minutes", "number", "Expected meeting duration", required=False, default=30),
            ],
            outputs=[
                SkillOutput("briefing", "object", "Complete meeting preparation package"),
            ],
            dependencies=[
                SkillDependency("tool", "calendar", reason="Check for conflicts, past related meetings"),
                SkillDependency("tool", "http", reason="Research attendees and topics"),
            ],
        )
        
        return Skill(metadata, {
            "execution_plan": [
                {"step": "context", "description": "Gather context from calendar, emails, related documents"},
                {"step": "attendees", "description": "Research attendee backgrounds and relationships"},
                {"step": "agenda", "description": "Draft proposed agenda based on likely objectives"},
                {"step": "materials", "description": "Prepare talking points, questions, and reference materials"},
                {"step": "strategy", "description": "Suggest participation strategy based on role and goals"},
            ],
        }, "composite")
    
    def _create_code_review_skill(self) -> Skill:
        """Create the code review skill."""
        metadata = SkillMetadata(
            name="review_code",
            version="1.0.0",
            description="Review code for bugs, style, security, and performance",
            complexity=SkillComplexity.MODERATE,
            tags=["code", "review", "quality", "security"],
            categories=["development", "quality-assurance"],
            when_to_use="Before committing, in PR reviews, when learning new codebase, for security audits",
            parameters=[
                SkillParameter("code", "string", "Code to review", required=True),
                SkillParameter("language", "string", "Programming language", required=True),
                SkillParameter("focus_areas", "array", "Aspects to emphasize: 'security', 'performance', 'style', 'bugs'", 
                              required=False, default=["bugs", "security"]),
                SkillParameter("context", "string", "What this code is supposed to do", required=False, default=""),
            ],
            outputs=[
                SkillOutput("findings", "array", "List of issues with severity, location, explanation, fix"),
                SkillOutput("summary", "object", "Overall assessment: quality_score, confidence, key_concerns"),
                SkillOutput("suggestions", "array", "Improvement opportunities not strictly issues"),
            ],
        )
        
        return Skill(metadata, {}, "llm")  # LLM-based implementation
    
    def _create_learning_skill(self) -> Skill:
        """Create the meta-skill for learning new skills."""
        metadata = SkillMetadata(
            name="learn_skill",
            version="1.0.0",
            description="Create a new skill from description, example, or demonstration",
            long_description="""
            The ultimate meta-skill: learns how to do new things and encodes them
            as reusable, documented, versioned skills. Can learn from:
            - Natural language description of desired behavior
            - Step-by-step demonstration (what I do)
            - Example input/output pairs (what I want)
            - Abstraction of existing skills (compose, specialize, generalize)
            """,
            complexity=SkillComplexity.SYSTEM,
            tags=["meta", "learning", "creation", "evolution"],
            categories=["system", "meta-cognitive"],
            when_to_use="When you need to automate something new, when current skills don't fit, to refine existing skills",
            when_not_to_use="For one-off tasks, when existing skill suffices, for tasks requiring human judgment",
            estimated_duration="2-10 minutes depending on complexity",
            parameters=[
                SkillParameter("method", "string", "Learning method: 'description', 'demonstration', 'example', 'abstraction'", 
                              required=True),
                SkillParameter("name", "string", "Name for the new skill", required=True),
                SkillParameter("input", "object", "Learning input based on method", required=True),
                SkillParameter("validate", "boolean", "Test skill with examples before saving", 
                              required=False, default=True),
            ],
            outputs=[
                SkillOutput("skill_created", "object", "Metadata of created skill"),
                SkillOutput("confidence", "number", "Estimated reliability 0-1"),
                SkillOutput("validation_results", "object", "Test results if validate=True"),
            ],
            dependencies=[
                SkillDependency("skill", "validate_skill", reason="Test newly created skills"),
            ],
        )
        
        implementation = {
            "learning_strategies": {
                "description": """
                    From natural language description:
                    1. Extract intent, parameters, expected behavior
                    2. Identify required tools and dependencies
                    3. Generate implementation plan or code
                    4. Create comprehensive metadata
                """,
                "demonstration": """
                    From step-by-step demonstration:
                    1. Record each step and decision point
                    2. Abstract patterns into reusable logic
                    3. Identify parameterizable components
                    4. Document implicit knowledge
                """,
                "example": """
                    From input/output examples:
                    1. Infer transformation logic
                    2. Identify edge cases and constraints
                    3. Generate implementation covering examples
                    4. Create additional test cases
                """,
                "abstraction": """
                    From existing skills:
                    1. Identify common patterns across skills
                    2. Extract reusable components
                    3. Create parameterized generalization
                    4. Maintain relationship to parent skills
                """,
            },
        }
        
        return Skill(metadata, implementation, "llm")
    
    def _load_user_skills(self) -> None:
        """Load skills from user skills directory."""
        if not self.skills_dir.exists():
            return
        
        for skill_file in self.skills_dir.glob("*.skill.json"):
            try:
                data = json.loads(skill_file.read_text())
                skill = Skill(
                    metadata=SkillMetadata.from_dict(data["metadata"]),
                    implementation=data.get("implementation"),
                    implementation_type=data.get("implementation_type", "code"),
                )
                self.register(skill)
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")
    
    def register(self, skill: Skill, is_builtin: bool = False) -> None:
        """Register a skill in the registry."""
        name = skill.name
        
        # Store in version history
        if name not in self._version_history:
            self._version_history[name] = []
        self._version_history[name].append(skill)
        
        # Update latest
        self._skills[name] = skill
        
        # Update indexes
        for category in skill.metadata.categories:
            self._categories.setdefault(category, set()).add(name)
        
        for tag in skill.metadata.tags:
            self._tags.setdefault(tag, set()).add(name)
        
        # Build search index
        text_to_index = f"{name} {skill.metadata.description} {' '.join(skill.metadata.tags)}"
        for word in set(text_to_index.lower().split()):
            self._search_index.setdefault(word, set()).add(name)
        
        source = "built-in" if is_builtin else "user"
        logger.debug(f"Registered skill '{name}' v{skill.metadata.version} ({source})")
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[Skill]:
        """Get a skill by name, optionally specific version."""
        if version:
            # Search version history
            for skill in self._version_history.get(name, []):
                if skill.metadata.version == version:
                    return skill
            return None
        return self._skills.get(name)
    
    def list_skills(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        complexity: Optional[SkillComplexity] = None,
    ) -> List[Skill]:
        """List skills matching criteria."""
        candidates = set(self._skills.keys())
        
        if category:
            candidates &= self._categories.get(category, set())
        
        if tag:
            candidates &= self._tags.get(tag, set())
        
        skills = [self._skills[name] for name in candidates]
        
        if complexity:
            skills = [s for s in skills if s.metadata.complexity == complexity]
        
        return skills
    
    def search(self, query: str) -> List[Tuple[Skill, float]]:
        """Search skills by relevance to query."""
        query_words = set(query.lower().split())
        
        # Score by word overlap
        scores: Dict[str, float] = {}
        for word in query_words:
            for skill_name in self._search_index.get(word, set()):
                scores[skill_name] = scores.get(skill_name, 0) + 1
        
        # Normalize by description length (shorter = more focused)
        results = []
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            skill = self._skills[name]
            # Boost exact name match
            if query.lower() in name.lower():
                score += 2
            # Boost high-confidence skills
            score *= skill.metadata.confidence_score
            results.append((skill, score))
        
        return sorted(results, key=lambda x: -x[1])
    
    def recommend(self, context: str, available_tools: Set[str]) -> List[Tuple[Skill, str]]:
        """Recommend skills based on context and available capabilities."""
        # Search for relevant skills
        candidates = self.search(context)[:10]
        
        # Filter by satisfiable dependencies
        available_skills = set(self._skills.keys())
        viable = []
        for skill, score in candidates:
            can_run, missing = skill.check_dependencies(available_tools, available_skills)
            if can_run:
                viable.append((skill, f"Relevance: {score:.1f}"))
            elif all(d.optional for d in skill.metadata.dependencies if not d.is_satisfied(available_tools, available_skills)):
                viable.append((skill, f"Partial (missing optional: {missing})"))
        
        return viable[:5]
    
    def get_dependency_graph(self, skill_name: str) -> Dict[str, Any]:
        """Build dependency tree for a skill."""
        skill = self._skills.get(skill_name)
        if not skill:
            return {}
        
        def build_tree(name: str, visited: Set[str]) -> Dict[str, Any]:
            if name in visited:
                return {"name": name, "circular": True}
            visited.add(name)
            
            s = self._skills.get(name)
            if not s:
                return {"name": name, "missing": True}
            
            deps = []
            for dep in s.metadata.dependencies:
                if dep.kind == "skill":
                    deps.append(build_tree(dep.name, visited.copy()))
            
            return {
                "name": name,
                "version": s.metadata.version,
                "dependencies": deps,
            }
        
        return build_tree(skill_name, set())


class SkillExecutor:
    """
    Executes skills with proper context, logging, and error handling.
    
    Provides sandboxed execution environment, dependency injection,
    and comprehensive execution tracing.
    """
    
    def __init__(
        self,
        agent: Agent,
        registry: SkillRegistry,
        guardian = None,
    ):
        self.agent = agent
        self.registry = registry
        self.guardian = guardian or get_guardian()
        self._execution_log: List[SkillExecutionRecord] = []
        self._max_log_size = 1000
        
        # Execution context stack for nested skill calls
        self._context_stack: List[Dict[str, Any]] = []
    
    async def execute(
        self,
        skill_name: str,
        inputs: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a skill with full logging and error handling.
        """
        skill = self.registry.get(skill_name)
        if not skill:
            return {
                "success": False,
                "error": f"Skill '{skill_name}' not found",
                "skill_name": skill_name,
            }
        
        # Create execution record
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        record = SkillExecutionRecord(
            execution_id=execution_id,
            skill_name=skill_name,
            skill_version=skill.metadata.version,
            started_at=datetime.now(),
            input_params=inputs,
        )
        
        # Validate inputs
        valid, errors = skill.validate_inputs(inputs)
        if not valid:
            record.success = False
            record.error_message = f"Input validation failed: {'; '.join(errors)}"
            record.completed_at = datetime.now()
            self._log_execution(record)
            return {
                "success": False,
                "error": record.error_message,
                "validation_errors": errors,
                "execution_id": execution_id,
            }
        
        # Check dependencies
        available_tools = set(self.agent.tools.keys()) if hasattr(self.agent, 'tools') else set()
        available_skills = set(self.registry._skills.keys())
        
        deps_ok, missing = skill.check_dependencies(available_tools, available_skills)
        if not deps_ok:
            return {
                "success": False,
                "error": f"Missing dependencies: {', '.join(missing)}",
                "execution_id": execution_id,
            }
        
        # Guardian pre-authorization for complex skills
        if skill.metadata.complexity in (SkillComplexity.COMPLEX, SkillComplexity.SYSTEM):
            allowed, reason, guard_event = self.guardian.check_tool_execution(
                f"skill:{skill_name}", inputs, "skill_executor", execution_context or {}
            )
            if not allowed:
                record.guardian_events.append(f"BLOCKED: {reason}")
                record.success = False
                record.error_message = f"Guardian blocked: {reason}"
                record.completed_at = datetime.now()
                self._log_execution(record)
                return {
                    "success": False,
                    "error": record.error_message,
                    "guardian_action": "block",
                    "execution_id": execution_id,
                }
        
        # Push context
        self._context_stack.append({
            "skill_name": skill_name,
            "execution_id": execution_id,
            "inputs": inputs,
        })
        
        # Execute based on implementation type
        try:
            if skill.implementation_type == "code":
                result = await self._execute_code_skill(skill, inputs, record)
            elif skill.implementation_type == "composite":
                result = await self._execute_composite_skill(skill, inputs, record)
            elif skill.implementation_type == "llm":
                result = await self._execute_llm_skill(skill, inputs, record)
            elif skill.implementation_type == "template":
                result = await self._execute_template_skill(skill, inputs, record)
            else:
                raise ValueError(f"Unknown implementation type: {skill.implementation_type}")
            
            # Update record with success
            record.success = True
            record.output_data = result
            record.completed_at = datetime.now()
            record.duration_seconds = (record.completed_at - record.started_at).total_seconds()
            
            # Update skill statistics
            skill.metadata.execution_count += 1
            # Simple running average of success rate
            skill.metadata.success_rate = (
                (skill.metadata.success_rate * (skill.metadata.execution_count - 1) + 1) 
                / skill.metadata.execution_count
            )
            skill.metadata.last_executed = datetime.now()
            
            return {
                "success": True,
                "result": result,
                "execution_id": execution_id,
                "duration_seconds": record.duration_seconds,
            }
            
        except Exception as e:
            record.success = False
            record.error_message = str(e)
            record.completed_at = datetime.now()
            record.duration_seconds = (record.completed_at - record.started_at).total_seconds()
            
            # Update skill statistics
            skill.metadata.execution_count += 1
            skill.metadata.success_rate = (
                skill.metadata.success_rate * (skill.metadata.execution_count - 1)
            ) / skill.metadata.execution_count
            
            logger.exception(f"Skill execution failed: {skill_name}")
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id,
                "duration_seconds": record.duration_seconds,
            }
        
        finally:
            # Pop context
            self._context_stack.pop()
            self._log_execution(record)
    
    async def _execute_code_skill(self, skill: Skill, inputs: Dict[str, Any], record: SkillExecutionRecord) -> Any:
        """Execute a Python code-based skill."""
        # For security, code skills would run in restricted environment
        # This is a simplified version - production would use sandboxing
        func = skill.implementation
        if callable(func):
            # Inject agent and tools as needed
            sig = inspect.signature(func)
            kwargs = {}
            if 'agent' in sig.parameters:
                kwargs['agent'] = self.agent
            if 'tools' in sig.parameters:
                kwargs['tools'] = self.agent.tools if hasattr(self.agent, 'tools') else {}
            if 'call_skill' in sig.parameters:
                kwargs['call_skill'] = self._make_skill_caller(record)
            
            result = func(**inputs, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        
        raise ValueError("Code skill implementation is not callable")
    
    async def _execute_composite_skill(self, skill: Skill, inputs: Dict[str, Any], record: SkillExecutionRecord) -> Any:
        """Execute a composite skill by orchestrating other skills/tools."""
        plan = skill.implementation.get("execution_plan", [])
        results = {}
        
        for step in plan:
            step_name = step["step"]
            description = step["description"]
            record.steps_executed.append(f"{step_name}: {description}")
            
            # Use agent's planning to execute step
            step_result = await self.agent.chat(
                user_id=f"skill:{skill.name}",
                message=f"Execute step '{step_name}': {description}. "
                       f"Context: inputs={inputs}, previous_results={results}",
                context={"skill_execution": True, "step": step_name},
            )
            
            if not step_result.get("success"):
                raise RuntimeError(f"Step '{step_name}' failed: {step_result.get('error')}")
            
            results[step_name] = step_result.get("response")
            record.tools_used.extend(step_result.get("tools_used", []))
        
        # Compile final output based on skill's output schema
        return {
            "steps_completed": list(results.keys()),
            "final_result": results.get(plan[-1]["step"]) if plan else None,
            "all_results": results,
        }
    
    async def _execute_llm_skill(self, skill: Skill, inputs: Dict[str, Any], record: SkillExecutionRecord) -> Any:
        """Execute an LLM-guided skill."""
        # Build comprehensive prompt from metadata
        system_prompt = self._build_skill_system_prompt(skill)
        
        # Construct user prompt from inputs
        user_prompt = f"Execute skill '{skill.name}' with parameters:\n"
        for param in skill.metadata.parameters:
            value = inputs.get(param.name)
            if value is not None:
                user_prompt += f"- {param.name}: {json.dumps(value)}\n"
        
        # Execute through agent
        result = await self.agent.chat(
            user_id=f"skill:{skill.name}",
            message=user_prompt,
            context={
                "skill_execution": True,
                "system_prompt_override": system_prompt,
            },
        )
        
        if not result.get("success"):
            raise RuntimeError(f"LLM skill execution failed: {result.get('error')}")
        
        # Parse and validate output against schema
        output_text = result.get("response", "")
        try:
            # Try to extract JSON
            parsed = self._extract_json(output_text)
            return parsed
        except:
            # Return as structured text
            return {"raw_output": output_text}
    
    async def _execute_template_skill(self, skill: Skill, inputs: Dict[str, Any], record: SkillExecutionRecord) -> Any:
        """Execute a template-based skill."""
        # Would use Jinja2 or similar
        template = skill.implementation
        # Simple string substitution for now
        result = template
        for key, value in inputs.items():
            result = result.replace(f"{{{{ {key} }}}}", str(value))
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return {"rendered": result}
    
    def _build_skill_system_prompt(self, skill: Skill) -> str:
        """Build system prompt for LLM skill execution."""
        lines = [
            f"You are executing the skill: {skill.name}",
            "",
            f"Description: {skill.metadata.description}",
            "",
            "Your task is to execute this skill correctly, following these steps:",
        ]
        
        if skill.implementation_type == "composite":
            plan = skill.implementation.get("execution_plan", [])
            for i, step in enumerate(plan, 1):
                lines.append(f"{i}. {step['step']}: {step['description']}")
        
        lines.extend([
            "",
            "Output Requirements:",
        ])
        for output in skill.metadata.outputs:
            lines.append(f"- {output.name} ({output.type}): {output.description}")
        
        if skill.metadata.examples:
            lines.extend(["", "Example:", json.dumps(skill.metadata.examples[0].expected_output, indent=2)])
        
        return '\n'.join(lines)
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text that may contain markdown or other content."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
            r'\[.*\]',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        raise ValueError("No valid JSON found in text")
    
    def _make_skill_caller(self, parent_record: SkillExecutionRecord) -> Callable:
        """Create a function for skills to call other skills."""
        async def call_skill(skill_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
            result = await self.execute(skill_name, inputs, {
                "parent_execution": parent_record.execution_id,
            })
            parent_record.skills_called.append(skill_name)
            return result
        
        return call_skill
    
    def _log_execution(self, record: SkillExecutionRecord) -> None:
        """Record execution in log."""
        self._execution_log.append(record)
        if len(self._execution_log) > self._max_log_size:
            self._execution_log.pop(0)
    
    def get_execution_history(
        self,
        skill_name: Optional[str] = None,
        since: Optional[datetime] = None,
        success_only: bool = False,
    ) -> List[SkillExecutionRecord]:
        """Query execution history."""
        results = self._execution_log
        
        if skill_name:
            results = [r for r in results if r.skill_name == skill_name]
        
        if since:
            results = [r for r in results if r.started_at >= since]
        
        if success_only:
            results = [r for r in results if r.success]
        
        return results
    
    def analyze_skill_performance(self, skill_name: str) -> Dict[str, Any]:
        """Analyze execution statistics for a skill."""
        records = [r for r in self._execution_log if r.skill_name == skill_name]
        
        if not records:
            return {"error": "No execution records found"}
        
        total = len(records)
        successful = sum(1 for r in records if r.success)
        durations = [r.duration_seconds for r in records if r.duration_seconds > 0]
        
        return {
            "total_executions": total,
            "success_count": successful,
            "failure_count": total - successful,
            "success_rate": successful / total,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "recent_errors": [r.error_message for r in records[-10:] if not r.success],
        }


class SkillLearner:
    """
    Creates new skills through learning from examples, descriptions, or demonstrations.
    
    This is the engine of LollmsBot's growth - turning experience into reusable capability.
    """
    
    def __init__(
        self,
        registry: SkillRegistry,
        executor: SkillExecutor,
    ):
        self.registry = registry
        self.executor = executor
    
    async def learn_from_description(
        self,
        name: str,
        description: str,
        example_inputs: List[Dict[str, Any]],
        expected_outputs: List[Dict[str, Any]],
        complexity_hint: Optional[SkillComplexity] = None,
    ) -> Skill:
        """
        Create a skill from natural language description and examples.
        """
        # Analyze to extract parameters
        parameters = self._infer_parameters(example_inputs)
        
        # Determine complexity
        complexity = complexity_hint or self._estimate_complexity(description, parameters)
        
        # Build metadata
        metadata = SkillMetadata(
            name=name,
            description=description,
            complexity=complexity,
            parameters=parameters,
            outputs=self._infer_outputs(expected_outputs),
            learning_method="description",
            examples=[
                SkillExample(
                    description=f"Example {i+1}",
                    input_params=inp,
                    expected_output=out,
                )
                for i, (inp, out) in enumerate(zip(example_inputs, expected_outputs))
            ],
        )
        
        # Generate implementation
        if complexity in (SkillComplexity.TRIVIAL, SkillComplexity.SIMPLE):
            implementation = await self._generate_code_implementation(metadata, example_inputs, expected_outputs)
            impl_type = "code"
        else:
            implementation = await self._generate_llm_implementation(metadata)
            impl_type = "llm"
        
        skill = Skill(metadata, implementation, impl_type)
        
        # Validate with examples
        validation_results = []
        for inp, expected in zip(example_inputs, expected_outputs):
            result = await self.executor.execute(name, inp)
            validation_results.append({
                "input": inp,
                "expected": expected,
                "actual": result,
                "match": self._outputs_match(expected, result.get("result", {})),
            })
        
        # Calculate confidence from validation
        matches = sum(1 for v in validation_results if v["match"])
        skill.metadata.confidence_score = matches / len(validation_results) if validation_results else 0.5
        
        # Register if confidence sufficient
        if skill.metadata.confidence_score >= 0.7:
            self.registry.register(skill)
            return skill
        else:
            # Return for refinement
            return skill  # Caller can decide to refine or discard
    
    async def learn_from_skill_composition(
        self,
        name: str,
        component_skills: List[str],
        data_flow: Dict[str, Any],  # How outputs feed into inputs
        description: str,
    ) -> Skill:
        """
        Create a new skill by composing existing skills.
        """
        # Verify all components exist
        missing = [s for s in component_skills if s not in self.registry._skills]
        if missing:
            raise ValueError(f"Unknown component skills: {missing}")
        
        # Build composite implementation
        implementation = {
            "component_skills": component_skills,
            "data_flow": data_flow,
            "execution_plan": [
                {"step": f"call_{skill}", "description": f"Execute {skill} with mapped inputs"}
                for skill in component_skills
            ],
        }
        
        # Infer parameters from first skill
        first_skill = self.registry.get(component_skills[0])
        parameters = first_skill.metadata.parameters if first_skill else []
        
        # Infer outputs from last skill
        last_skill = self.registry.get(component_skills[-1])
        outputs = last_skill.metadata.outputs if last_skill else []
        
        metadata = SkillMetadata(
            name=name,
            description=description,
            complexity=SkillComplexity.COMPLEX,
            parameters=parameters,
            outputs=outputs,
            dependencies=[
                SkillDependency("skill", s, reason=f"Composed component: {s}")
                for s in component_skills
            ],
            learning_method="composition",
            parent_skill=component_skills[0] if len(component_skills) == 1 else None,
        )
        
        skill = Skill(metadata, implementation, "composite")
        self.registry.register(skill)
        return skill
    
    def _infer_parameters(self, examples: List[Dict[str, Any]]) -> List[SkillParameter]:
        """Infer parameter schema from examples."""
        if not examples:
            return []
        
        # Find all keys across examples
        all_keys = set()
        for ex in examples:
            all_keys.update(ex.keys())
        
        parameters = []
        for key in sorted(all_keys):
            # Determine type from values
            values = [ex.get(key) for ex in examples if key in ex]
            types_found = set(type(v).__name__ for v in values if v is not None)
            
            type_map = {
                "str": "string",
                "int": "number",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            
            param_type = "string"  # default
            for t in types_found:
                if t in type_map:
                    param_type = type_map[t]
                    break
            
            # Check if always present
            required = all(key in ex for ex in examples)
            
            # Find examples
            example_values = [v for v in values if v is not None][:3]
            
            parameters.append(SkillParameter(
                name=key,
                type=param_type,
                description=f"Parameter: {key}",
                required=required,
                examples=example_values,
            ))
        
        return parameters
    
    def _infer_outputs(self, examples: List[Dict[str, Any]]) -> List[SkillOutput]:
        """Infer output schema from examples."""
        if not examples:
            return []
        
        # Simplified: assume flat structure
        all_keys = set()
        for ex in examples:
            all_keys.update(ex.keys())
        
        return [
            SkillOutput(name=k, type="object", description=f"Output: {k}")
            for k in sorted(all_keys)
        ]
    
    def _estimate_complexity(self, description: str, parameters: List[SkillParameter]) -> SkillComplexity:
        """Estimate complexity from description and parameter count."""
        desc_lower = description.lower()
        
        # Check for complexity indicators
        if any(w in desc_lower for w in ["simple", "trivial", "basic", "just", "only"]):
            return SkillComplexity.SIMPLE
        
        if any(w in desc_lower for w in ["orchestrate", "workflow", "multi-step", "complex"]):
            return SkillComplexity.COMPLEX
        
        if len(parameters) > 5:
            return SkillComplexity.MODERATE
        
        return SkillComplexity.SIMPLE
    
    async def _generate_code_implementation(
        self,
        metadata: SkillMetadata,
        examples: List[Dict[str, Any]],
        expected: List[Dict[str, Any]],
    ) -> str:
        """Generate Python code for simple skills."""
        # In production, this would use LLM code generation
        # For now, return a template
        param_list = ", ".join(p.name for p in metadata.parameters)
        
        code = f'''
async def {metadata.name}({param_list}, agent=None, tools=None, call_skill=None):
    """
    {metadata.description}
    
    Generated from examples with confidence: {metadata.confidence_score}
    """
    # TODO: Implement based on examples
    # Examples provided: {len(examples)}
    
    result = {{}}
    
    # Use available tools
    if "filesystem" in tools:
        fs = tools["filesystem"]
        # Implementation here
    
    return result
'''
        return code
    
    async def _generate_llm_implementation(self, metadata: SkillMetadata) -> Dict[str, Any]:
        """Generate LLM-guided implementation for complex skills."""
        return {
            "system_prompt": f"You are executing: {metadata.description}",
            "execution_guidance": "Break into steps, use tools as needed, validate outputs",
        }
    
    def _outputs_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if actual output matches expected structure."""
        # Simplified: check key overlap
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        
        # Allow extra keys in actual, but require all expected keys
        return expected_keys <= actual_keys


# Global registry access
_skill_registry: Optional[SkillRegistry] = None
_skill_executor: Optional[SkillExecutor] = None

def get_skill_registry() -> SkillRegistry:
    """Get or create global skill registry."""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry

def get_skill_executor(agent: Optional[Agent] = None) -> SkillExecutor:
    """Get or create global skill executor."""
    global _skill_executor
    if _skill_executor is None:
        if agent is None:
            raise ValueError("Agent required for skill executor initialization")
        _skill_executor = SkillExecutor(agent, get_skill_registry())
    return _skill_executor
