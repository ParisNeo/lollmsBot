"""
CRM Module - Contact Relationship Management for lollmsBot

Provides contact ingestion from Gmail/calendar, deduplication,
timeline tracking, and natural language querying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class ContactRole(Enum):
    """Classification of contact relationships."""
    UNKNOWN = "unknown"
    CLIENT = "client"
    PARTNER = "partner"
    VENDOR = "vendor"
    INVESTOR = "investor"
    MEDIA = "media"
    EMPLOYEE = "employee"
    PROSPECT = "prospect"
    NEWSLETTER = "newsletter"  # To be filtered out
    COLD_OUTREACH = "cold_outreach"  # To be filtered out


@dataclass
class ContactInteraction:
    """A single interaction with a contact."""
    timestamp: datetime
    interaction_type: str  # "email", "meeting", "call", "note"
    content_summary: str
    source_id: str  # Email ID, meeting ID, etc.
    sentiment: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    importance: float = 1.0


@dataclass
class Contact:
    """A contact in the CRM system."""
    contact_id: str
    name: str
    email: str
    role: ContactRole = ContactRole.UNKNOWN
    company: Optional[str] = None
    title: Optional[str] = None
    
    # Interaction history
    interactions: List[ContactInteraction] = field(default_factory=list)
    last_touch: Optional[datetime] = None
    first_touch: Optional[datetime] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    importance_score: float = 1.0  # Calculated from interaction frequency/quality
    
    # Source tracking
    sources: Set[str] = field(default_factory=set)  # "gmail", "calendar", "manual"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "name": self.name,
            "email": self.email,
            "role": self.role.value,
            "company": self.company,
            "title": self.title,
            "interaction_count": len(self.interactions),
            "last_touch": self.last_touch.isoformat() if self.last_touch else None,
            "first_touch": self.first_touch.isoformat() if self.first_touch else None,
            "tags": list(self.tags),
            "importance_score": self.importance_score,
            "sources": list(self.sources),
        }
    
    def get_timeline_summary(self, limit: int = 5) -> str:
        """Get recent interaction summary."""
        recent = sorted(self.interactions, key=lambda x: x.timestamp, reverse=True)[:limit]
        lines = [f"Recent interactions with {self.name}:"]
        for i in recent:
            lines.append(f"  [{i.timestamp.strftime('%Y-%m-%d')}] {i.interaction_type}: {i.content_summary[:60]}...")
        return "\n".join(lines)


class CRMManager:
    """
    Manages contact relationships with automatic ingestion
    from Gmail, calendar, and manual entries.
    """
    
    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager
        self._contacts: Dict[str, Contact] = {}  # email -> Contact
        self._initialized = False
    
    async def initialize(self) -> None:
        """Load existing contacts from memory."""
        if self._initialized:
            return
        
        # Search for existing CRM data in memory
        if self._memory:
            results = await self._memory.search_ems(
                query="crm contact",
                chunk_types=[MemoryChunkType.FACT],
                limit=100,
            )
            
            for chunk, score in results:
                if "crm_contact:" in (chunk.source or ""):
                    try:
                        import json
                        data = json.loads(chunk.decompress_content())
                        contact = self._dict_to_contact(data)
                        self._contacts[contact.email.lower()] = contact
                    except Exception as e:
                        logger.warning(f"Failed to load contact from chunk: {e}")
        
        self._initialized = True
        logger.info(f"CRM initialized with {len(self._contacts)} contacts")
    
    def _dict_to_contact(self, data: Dict[str, Any]) -> Contact:
        """Reconstruct contact from dictionary."""
        contact = Contact(
            contact_id=data.get("contact_id", ""),
            name=data.get("name", ""),
            email=data.get("email", ""),
            role=ContactRole(data.get("role", "unknown")),
            company=data.get("company"),
            title=data.get("title"),
            tags=set(data.get("tags", [])),
            sources=set(data.get("sources", [])),
            importance_score=data.get("importance_score", 1.0),
        )
        
        # Reconstruct interactions
        for i_data in data.get("interactions", []):
            contact.interactions.append(ContactInteraction(
                timestamp=datetime.fromisoformat(i_data["timestamp"]),
                interaction_type=i_data["interaction_type"],
                content_summary=i_data["content_summary"],
                source_id=i_data["source_id"],
                sentiment=i_data.get("sentiment"),
                topics=i_data.get("topics", []),
                importance=i_data.get("importance", 1.0),
            ))
        
        if data.get("last_touch"):
            contact.last_touch = datetime.fromisoformat(data["last_touch"])
        if data.get("first_touch"):
            contact.first_touch = datetime.fromisoformat(data["first_touch"])
        
        return contact
    
    async def ingest_email(self, email_data: Dict[str, Any]) -> Optional[Contact]:
        """
        Process an email and update/create contact.
        
        Filters out newsletters and cold outreach automatically.
        """
        from email.utils import parseaddr
        
        sender_raw = email_data.get("from", "")
        sender_name, sender_email = parseaddr(sender_raw)
        
        if not sender_email:
            return None
        
        # Check for filter patterns
        subject = email_data.get("subject", "").lower()
        body = email_data.get("body", "").lower()
        
        # Filter patterns
        filter_indicators = [
            "unsubscribe", "newsletter", "marketing", "promotional",
            "no-reply", "noreply", "donotreply", "automated",
        ]
        
        if any(ind in sender_email.lower() or ind in subject or ind in body[:500] 
               for ind in filter_indicators):
            logger.debug(f"Filtered email from {sender_email} (newsletter/automated)")
            return None
        
        # Check for cold outreach patterns
        cold_patterns = [
            "quick question", "reaching out", "opportunity", "partnership",
            "collaboration", "guest post", "sponsored", "advertise",
        ]
        
        is_cold = any(p in subject.lower() or p in body[:1000].lower() for p in cold_patterns)
        
        # Get or create contact
        email_key = sender_email.lower()
        contact = self._contacts.get(email_key)
        
        if contact is None:
            contact = Contact(
                contact_id=f"contact_{email_key.replace('@', '_')}",
                name=sender_name or sender_email.split("@")[0],
                email=sender_email,
                first_touch=datetime.now(),
                sources={"gmail"},
            )
            self._contacts[email_key] = contact
        
        # Add interaction
        interaction = ContactInteraction(
            timestamp=email_data.get("date", datetime.now()),
            interaction_type="email",
            content_summary=email_data.get("subject", "No subject"),
            source_id=email_data.get("message_id", ""),
            topics=self._extract_topics(subject + " " + body[:500]),
            importance=0.5 if is_cold else 1.0,
        )
        
        contact.interactions.append(interaction)
        contact.last_touch = interaction.timestamp
        contact.sources.add("gmail")
        
        # Classify role using simple heuristics (would use AI in production)
        contact.role = self._classify_role(contact, subject, body[:1000])
        
        # Update importance score
        contact.importance_score = self._calculate_importance(contact)
        
        # Persist to memory
        await self._persist_contact(contact)
        
        return contact
    
    async def ingest_calendar_event(self, event_data: Dict[str, Any]) -> List[Contact]:
        """Process calendar event and update participant contacts."""
        participants = event_data.get("participants", [])
        created_contacts = []
        
        for participant in participants:
            email = participant.get("email", "").lower()
            if not email or email.endswith("@resource.calendar.google.com"):
                continue  # Skip resource rooms
            
            contact = self._contacts.get(email)
            
            if contact is None:
                contact = Contact(
                    contact_id=f"contact_{email.replace('@', '_')}",
                    name=participant.get("name", email.split("@")[0]),
                    email=email,
                    first_touch=datetime.now(),
                    sources={"calendar"},
                )
                self._contacts[email] = contact
            
            # Add meeting interaction
            interaction = ContactInteraction(
                timestamp=event_data.get("start_time", datetime.now()),
                interaction_type="meeting",
                content_summary=event_data.get("summary", "Meeting"),
                source_id=event_data.get("event_id", ""),
                topics=self._extract_topics(event_data.get("description", "")),
                importance=2.0,  # Meetings are higher importance
            )
            
            contact.interactions.append(interaction)
            contact.last_touch = interaction.timestamp
            contact.sources.add("calendar")
            contact.importance_score = self._calculate_importance(contact)
            
            await self._persist_contact(contact)
            created_contacts.append(contact)
        
        return created_contacts
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple keyword extraction (would use NLP in production)
        words = text.lower().split()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through",
                     "during", "before", "after", "above", "below", "between",
                     "under", "again", "further", "then", "once", "here", "there",
                     "when", "where", "why", "how", "all", "each", "few", "more",
                     "most", "other", "some", "such", "no", "nor", "not", "only",
                     "own", "same", "so", "than", "too", "very", "just", "and",
                     "but", "if", "or", "because", "until", "while", "about",
                     "against", "between", "into", "through", "during", "before",
                     "after", "above", "below", "up", "down", "out", "off", "over",
                     "under", "again", "further", "then", "once"}
        
        # Find capitalized phrases and important words
        topics = []
        for word in words:
            clean = word.strip(".,;:!?()[]{}\"'")
            if len(clean) > 3 and clean not in stop_words:
                topics.append(clean)
        
        # Return unique topics, limited
        return list(set(topics))[:5]
    
    def _classify_role(self, contact: Contact, subject: str, body: str) -> ContactRole:
        """Classify contact role based on interaction content."""
        text = (subject + " " + body).lower()
        
        # Check for role indicators
        if any(w in text for w in ["sponsor", "sponsorship", "advertise", "ad deal", "brand deal"]):
            return ContactRole.CLIENT
        elif any(w in text for w in ["investor", "funding", "investment", "vc", "angel"]):
            return ContactRole.INVESTOR
        elif any(w in text for w in ["partner", "collaboration", "collab", "joint"]):
            return ContactRole.PARTNER
        elif any(w in text for w in ["press", "media", "interview", "journalist", "reporter"]):
            return ContactRole.MEDIA
        elif any(w in text for w in ["hiring", "job", "position", "candidate", "recruit"]):
            return ContactRole.EMPLOYEE
        
        # Check email domain
        domain = contact.email.split("@")[-1].lower()
        if domain in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]:
            return ContactRole.PROSPECT  # Personal email = likely prospect
        
        return ContactRole.UNKNOWN
    
    def _calculate_importance(self, contact: Contact) -> float:
        """Calculate importance score based on interaction history."""
        base_score = 1.0
        
        # Frequency bonus
        interaction_count = len(contact.interactions)
        if interaction_count > 10:
            base_score += 2.0
        elif interaction_count > 5:
            base_score += 1.0
        
        # Recency bonus
        if contact.last_touch:
            days_since = (datetime.now() - contact.last_touch).days
            if days_since < 7:
                base_score += 1.5
            elif days_since < 30:
                base_score += 0.5
        
        # Role multiplier
        role_multipliers = {
            ContactRole.CLIENT: 2.0,
            ContactRole.INVESTOR: 1.8,
            ContactRole.PARTNER: 1.5,
            ContactRole.MEDIA: 1.3,
            ContactRole.EMPLOYEE: 1.2,
            ContactRole.VENDOR: 1.0,
            ContactRole.PROSPECT: 0.8,
            ContactRole.UNKNOWN: 0.5,
        }
        
        return base_score * role_multipliers.get(contact.role, 1.0)
    
    async def _persist_contact(self, contact: Contact) -> None:
        """Save contact to memory system."""
        if not self._memory:
            return
        
        import json
        
        # Serialize contact
        data = {
            "contact_id": contact.contact_id,
            "name": contact.name,
            "email": contact.email,
            "role": contact.role.value,
            "company": contact.company,
            "title": contact.title,
            "interactions": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "interaction_type": i.interaction_type,
                    "content_summary": i.content_summary,
                    "source_id": i.source_id,
                    "sentiment": i.sentiment,
                    "topics": i.topics,
                    "importance": i.importance,
                }
                for i in contact.interactions
            ],
            "last_touch": contact.last_touch.isoformat() if contact.last_touch else None,
            "first_touch": contact.first_touch.isoformat() if contact.first_touch else None,
            "tags": list(contact.tags),
            "sources": list(contact.sources),
            "importance_score": contact.importance_score,
        }
        
        # Store in memory
        await self._memory.store_in_ems(
            content=json.dumps(data),
            chunk_type=MemoryChunkType.FACT,
            importance=contact.importance_score,
            tags=["crm", "contact", contact.role.value, f"email_{contact.email}"],
            summary=f"CRM: {contact.name} ({contact.email}) - {contact.role.value}, {len(contact.interactions)} interactions",
            load_hints=[contact.name, contact.email, contact.company or "", contact.role.value],
            source=f"crm_contact:{contact.contact_id}",
        )
    
    async def get_meeting_prep(self, contact_email: str, meeting_topic: Optional[str] = None) -> str:
        """
        Generate meeting preparation summary for a contact.
        This is the key feature shown in the SimplifiedAgant video.
        """
        contact = self._contacts.get(contact_email.lower())
        
        if not contact:
            return f"No contact found with email: {contact_email}"
        
        lines = [
            f"## Meeting Prep: {contact.name}",
            f"",
            f"**Role:** {contact.role.value}",
            f"**Company:** {contact.company or 'Unknown'}",
            f"**Title:** {contact.title or 'Unknown'}",
            f"**Total Interactions:** {len(contact.interactions)}",
            f"",
            f"### Recent Conversation History",
            contact.get_timeline_summary(5),
            f"",
        ]
        
        # Add topic-specific context if provided
        if meeting_topic:
            lines.extend([
                f"### Relevant to: {meeting_topic}",
                f"[AI would search for topic-related past discussions here]",
                f"",
            ])
        
        # Add suggested talking points based on role
        lines.append("### Suggested Talking Points")
        
        if contact.role == ContactRole.CLIENT:
            lines.extend([
                "- Review previous campaign performance",
                "- Discuss upcoming content calendar",
                "- Address any pending deliverables",
            ])
        elif contact.role == ContactRole.INVESTOR:
            lines.extend([
                "- Share recent channel growth metrics",
                "- Discuss business development updates",
                "- Address any due diligence items",
            ])
        elif contact.role == ContactRole.PARTNER:
            lines.extend([
                "- Review collaboration timeline",
                "- Discuss mutual promotion opportunities",
                "- Align on content themes",
            ])
        else:
            lines.extend([
                "- Reference last conversation topic",
                "- Ask about their current priorities",
                "- Offer value based on their role",
            ])
        
        return "\n".join(lines)
    
    async def query_contacts(self, query: str) -> List[Dict[str, Any]]:
        """Natural language query against contacts."""
        # Simple implementation - would use semantic search in production
        results = []
        query_lower = query.lower()
        
        for contact in self._contacts.values():
            score = 0
            
            # Name match
            if query_lower in contact.name.lower():
                score += 10
            
            # Company match
            if contact.company and query_lower in contact.company.lower():
                score += 8
            
            # Email match
            if query_lower in contact.email.lower():
                score += 5
            
            # Role match
            if query_lower in contact.role.value:
                score += 6
            
            # Topic match in interactions
            for interaction in contact.interactions:
                if any(query_lower in t for t in interaction.topics):
                    score += 3
                    break
            
            if score > 0:
                result = contact.to_dict()
                result["query_score"] = score
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: -x["query_score"])
        return results[:10]  # Top 10
    
    async def get_daily_digest(self) -> str:
        """Generate daily CRM update (runs via cron/heartbeat)."""
        # New contacts in last 24 hours
        recent_contacts = [
            c for c in self._contacts.values()
            if c.first_touch and (datetime.now() - c.first_touch).days < 1
        ]
        
        # Contacts with recent interactions
        active_contacts = [
            c for c in self._contacts.values()
            if c.last_touch and (datetime.now() - c.last_touch).days < 7
        ]
        
        lines = [
            "# 📊 CRM Daily Digest",
            "",
            f"**New contacts (24h):** {len(recent_contacts)}",
            f"**Active contacts (7d):** {len(active_contacts)}",
            f"**Total contacts:** {len(self._contacts)}",
            "",
        ]
        
        if recent_contacts:
            lines.append("## New Contacts Today")
            for c in recent_contacts:
                lines.append(f"- **{c.name}** ({c.email}) - {c.role.value}")
            lines.append("")
        
        if active_contacts:
            lines.append("## Recent Activity")
            for c in sorted(active_contacts, key=lambda x: x.last_touch or datetime.min, reverse=True)[:5]:
                lines.append(f"- **{c.name}**: Last contact {c.last_touch.strftime('%Y-%m-%d') if c.last_touch else 'unknown'}")
            lines.append("")
        
        return "\n".join(lines)


# Import for type hints
from lollmsbot.agent.rlm.models import MemoryChunkType


__all__ = [
    "CRMManager",
    "Contact",
    "ContactInteraction",
    "ContactRole",
]
