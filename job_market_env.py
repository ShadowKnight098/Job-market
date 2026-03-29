"""
Job Application Strategy Environment — OpenEnv compliant
Implements step() / reset() / state() API with typed models.

v1.1 — Added rejection reasons + skill recommendations engine.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from typing import Any


# ─────────────────────────────────────────────
# Skill & Rejection Knowledge Base
# ─────────────────────────────────────────────

# Skills required at each company tier
SKILL_CURRICULUM = {
    0: {  # Startup
        "name": "Startup",
        "required_skills": ["Python basics", "Git & version control", "REST APIs", "SQL fundamentals"],
        "nice_to_have":    ["Docker basics", "Any frontend framework", "Cloud basics (AWS/GCP free tier)"],
        "min_skill_level": 1,
    },
    1: {  # Mid-size
        "name": "Mid-size company",
        "required_skills": ["Data structures & algorithms", "System design basics", "Testing & CI/CD",
                            "OOP design patterns", "Database optimization"],
        "nice_to_have":    ["Kubernetes", "Message queues (Kafka/RabbitMQ)", "Machine learning basics"],
        "min_skill_level": 3,
    },
    2: {  # FAANG
        "name": "FAANG / Big Tech",
        "required_skills": ["Advanced DSA (LeetCode hard)", "Distributed systems design",
                            "Low-level optimization", "System design at scale", "Behavioral interviews (STAR)"],
        "nice_to_have":    ["Open source contributions", "Research papers", "Patents or notable projects",
                            "Leadership experience"],
        "min_skill_level": 5,
    },
}

# Rejection reasons indexed by (company_difficulty, skills_level)
# Each entry: list of possible rejection messages
REJECTION_REASONS = {
    # Startup rejections
    (0, 0): ["No coding experience shown on resume.",
             "Could not complete the basic coding screen.",
             "Profile lacked any practical projects."],
    (0, 1): ["Resume had no real-world projects to demonstrate skills.",
             "Basic coding test passed but take-home assignment was incomplete.",
             "Lacked experience with version control (Git)."],
    (0, 2): ["Stronger candidates with more project experience were selected.",
             "No portfolio or GitHub profile to review.",
             "Communication skills in the interview need improvement."],
    (0, 3): ["Another candidate had domain-specific startup experience.",
             "Culture fit concerns raised after interview.",
             "Salary expectations didn't align with the startup's budget."],
    (0, 4): ["Overqualified — startup worried about retention.",
             "Team decided to hire a more junior candidate to train.",
             "Role was filled internally at the last minute."],

    # Mid-size rejections
    (1, 0): ["Failed the initial coding screen completely.",
             "Resume did not meet minimum requirements.",
             "No relevant technical skills found in application."],
    (1, 1): ["Couldn't solve basic DSA problems (arrays, strings).",
             "No experience with testing or CI/CD pipelines.",
             "Lacked understanding of OOP principles."],
    (1, 2): ["Struggled with medium-level LeetCode problems in the interview.",
             "System design concepts were weak.",
             "Could not explain time/space complexity of solutions."],
    (1, 3): ["Solved coding problems but system design round was below bar.",
             "Good technical skills but team collaboration answers were vague.",
             "Database optimization knowledge was insufficient."],
    (1, 4): ["Very close — lost to a candidate with more leadership experience.",
             "Technical skills were strong but behavioral round fell short.",
             "Another candidate had directly relevant industry experience."],

    # FAANG rejections
    (2, 0): ["Couldn't pass the online assessment (OA).",
             "Resume was auto-filtered — no relevant experience.",
             "Failed the recruiter screen — no technical depth shown."],
    (2, 1): ["Online assessment score too low (solved 1/4 problems).",
             "No evidence of working on large-scale systems.",
             "DSA fundamentals not solid enough for FAANG bar."],
    (2, 2): ["Solved easy/medium Leetcode but failed the hard problems.",
             "System design round: couldn't design a scalable distributed system.",
             "Behavioral round: answers lacked leadership and ownership examples."],
    (2, 3): ["Passed coding rounds but failed system design (scale, consistency tradeoffs).",
             "Strong engineer but bar for senior role requires more depth.",
             "Solved problems but explanations were unclear to the panel."],
    (2, 4): ["Very competitive pool — top 5% but not top 1% this cycle.",
             "Passed all rounds but headcount was frozen after offer stage.",
             "Minor gaps in distributed systems depth compared to hired candidate."],
}

# Skill recommendations: what to learn given (company_difficulty, skills_level)
SKILL_RECOMMENDATIONS = {
    # Startup recommendations
    (0, 0): {
        "priority": ["Learn Python or JavaScript — pick one and build something",
                     "Complete a beginner Git tutorial (freeCodeCamp or Atlassian)"],
        "resources": ["freeCodeCamp.org (free)", "CS50 by Harvard (free on edX)", "The Odin Project (free)"],
        "timeline": "2–4 weeks to be ready for a startup application",
    },
    (0, 1): {
        "priority": ["Build 2–3 projects and put them on GitHub",
                     "Learn REST API design — build a simple CRUD app"],
        "resources": ["Roadmap.sh — Backend roadmap", "FastAPI tutorial (official docs)", "SQLZoo for SQL practice"],
        "timeline": "3–5 weeks to significantly improve your chances",
    },
    (0, 2): {
        "priority": ["Add a deployed project to your portfolio (use Vercel/Render free tier)",
                     "Improve your GitHub — daily commits, clean READMEs"],
        "resources": ["Render.com / Vercel (free deployment)", "GitHub Pages for portfolio",
                      "Pramp.com for mock interviews (free)"],
        "timeline": "2–3 weeks for quick wins",
    },
    (0, 3): {
        "priority": ["Research the startup's tech stack and tailor your resume",
                     "Prepare 3 strong behavioral stories (STAR format)"],
        "resources": ["Glassdoor for company research", "Levels.fyi for salary data",
                      "YouTube: 'STAR method interview' by Jeff H Sipe"],
        "timeline": "1 week of targeted prep",
    },
    (0, 4): {
        "priority": ["Frame yourself as a builder, not just an executor — show initiative",
                     "Negotiate confidently — startups often have flexibility"],
        "resources": ["'Cracking the PM Interview' for startup culture tips",
                      "Patrick McKenzie on salary negotiation (blog post)"],
        "timeline": "Soft skills focused — 1–2 weeks",
    },

    # Mid-size recommendations
    (1, 0): {
        "priority": ["Start DSA from scratch: arrays, strings, hashmaps",
                     "Solve 20 easy LeetCode problems before applying again"],
        "resources": ["NeetCode.io (free structured DSA)", "LeetCode easy problems",
                      "Abdul Bari's DSA course on YouTube (free)"],
        "timeline": "4–6 weeks of consistent practice",
    },
    (1, 1): {
        "priority": ["Focus on LeetCode: 50 easy + 30 medium problems",
                     "Learn and practice writing unit tests (pytest/jest)"],
        "resources": ["NeetCode 150 list", "TestDriven.io for Python testing",
                      "Fireship.io for concise tech tutorials"],
        "timeline": "4–6 weeks",
    },
    (1, 2): {
        "priority": ["Study system design basics: load balancers, caching, databases",
                     "Practice explaining your code out loud — record yourself"],
        "resources": ["'System Design Interview' by Alex Xu (book)", "Grokking System Design (free on GitHub)",
                      "interviewing.io for anonymous mock interviews"],
        "timeline": "6–8 weeks for solid system design foundation",
    },
    (1, 3): {
        "priority": ["Deep-dive system design: CAP theorem, consistent hashing, message queues",
                     "Prepare specific STAR stories for leadership, conflict, failure"],
        "resources": ["ByteByteGo newsletter (free tier)", "Designing Data-Intensive Applications (DDIA)",
                      "Exponent.fm behavioral interview prep"],
        "timeline": "3–4 weeks focused prep",
    },
    (1, 4): {
        "priority": ["Polish behavioral answers — quantify your impact with numbers",
                     "Research the company's engineering blog and reference it in the interview"],
        "resources": ["Company engineering blogs (Netflix Tech, Uber Engineering, etc.)",
                      "Levels.fyi for comp negotiation", "LinkedIn for referrals"],
        "timeline": "1–2 weeks final polish",
    },

    # FAANG recommendations
    (2, 0): {
        "priority": ["FAANG is not the right target yet — aim for startups first",
                     "Build fundamentals: DSA, OOP, databases from scratch"],
        "resources": ["NeetCode.io", "CS50 Harvard", "MIT OpenCourseWare 6.006 (Algorithms)"],
        "timeline": "6–12 months of building before targeting FAANG",
    },
    (2, 1): {
        "priority": ["Solve LeetCode: complete NeetCode 150 (all easy + medium)",
                     "Study time & space complexity deeply — Big O must be second nature"],
        "resources": ["NeetCode 150", "LeetCode Explore cards", "AlgoExpert (paid but worth it)"],
        "timeline": "3–4 months of daily practice",
    },
    (2, 2): {
        "priority": ["Attempt LeetCode hard problems daily — aim for 20+ solved",
                     "Study system design deeply: design Twitter, YouTube, Uber from scratch"],
        "resources": ["'Designing Data-Intensive Applications' — must read",
                      "System Design Primer (GitHub, free)", "LeetCode hard top 100"],
        "timeline": "2–3 months intensive prep",
    },
    (2, 3): {
        "priority": ["Master distributed systems: Raft, Paxos, eventual consistency",
                     "Build a side project that handles real scale (or simulate it)"],
        "resources": ["MIT 6.824 Distributed Systems (free lectures + labs on YouTube)",
                      "Google's Bigtable, Spanner, MapReduce papers (free PDFs)",
                      "interviewing.io for FAANG-level mock interviews"],
        "timeline": "2–3 months for distributed systems mastery",
    },
    (2, 4): {
        "priority": ["You're close — focus on consistency under pressure in interviews",
                     "Get a referral — it significantly improves your chances at FAANG"],
        "resources": ["Blind app for FAANG insider tips", "LinkedIn for referral connections",
                      "Exponent for behavioral prep at senior level"],
        "timeline": "2–4 weeks of final preparation + referral outreach",
    },
}


def get_rejection_reason(company_difficulty: int, skills_level: int) -> str:
    """Return a realistic rejection reason based on company tier and skill level."""
    key = (company_difficulty, min(skills_level, 4))
    reasons = REJECTION_REASONS.get(key, ["Application was not selected this time."])
    return random.choice(reasons)


def get_skill_recommendations(company_difficulty: int, skills_level: int) -> dict:
    """Return targeted skill recommendations for the given state."""
    key = (company_difficulty, min(skills_level, 4))
    return SKILL_RECOMMENDATIONS.get(key, {
        "priority":  ["Keep building projects and applying consistently."],
        "resources": ["LeetCode", "freeCodeCamp", "YouTube tutorials"],
        "timeline":  "Varies",
    })


# ─────────────────────────────────────────────
# Typed State Model
# ─────────────────────────────────────────────

@dataclass
class JobMarketState:
    skills_level: int          # 0–5  (0=no skills, 5=expert)
    applications_left: int     # 0–10
    company_difficulty: int    # 0=startup, 1=mid-size, 2=FAANG
    shortlisted_count: int     # how many times agent got shortlisted
    upskill_count: int         # how many times agent chose to upskill
    wasted_count: int          # failed application attempts
    episode_step: int          # current step number
    done: bool                 # episode ended

    def to_dict(self) -> dict:
        return asdict(self)

    def to_tuple(self) -> tuple:
        """Compact tuple for Q-table indexing."""
        return (
            self.skills_level,
            self.applications_left,
            self.company_difficulty,
        )


# ─────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────

ACTIONS = {
    0: "apply_small_company",
    1: "apply_big_company",
    2: "upskill",
}

ACTION_DESCRIPTIONS = {
    0: "Apply to a startup/small company (higher success at low skill)",
    1: "Apply to a big/FAANG company (higher reward, harder to get in)",
    2: "Upskill (+1 skill level, costs no application, +0.5 reward)",
}


# ─────────────────────────────────────────────
# Core Environment
# ─────────────────────────────────────────────

class JobMarketEnv:
    """
    OpenEnv-compliant job application strategy environment.

    Observation space:
        skills_level        int  [0, 5]
        applications_left   int  [0, 10]
        company_difficulty  int  {0, 1, 2}

    Action space:
        0 → apply_small_company
        1 → apply_big_company
        2 → upskill

    Reward:
        +1.0  →  shortlisted (success)
        +0.5  →  upskilled successfully
        -1.0  →  application failed (wasted attempt)
        -0.2  →  upskill at max level (wasted time)
        +1.5  →  BONUS: shortlisted at a big company (harder = higher reward)
    """

    metadata = {
        "env_id": "JobMarketStrategy-v1",
        "version": "1.0.0",
        "author": "OpenEnv Hackathon Submission",
        "observation_space": {
            "skills_level": {"type": "int", "range": [0, 5]},
            "applications_left": {"type": "int", "range": [0, 10]},
            "company_difficulty": {"type": "int", "values": [0, 1, 2]},
            "shortlisted_count": {"type": "int", "range": [0, 10]},
            "upskill_count": {"type": "int", "range": [0, 10]},
            "wasted_count": {"type": "int", "range": [0, 10]},
            "episode_step": {"type": "int", "range": [0, 10]},
            "done": {"type": "bool"},
        },
        "action_space": {
            "type": "discrete",
            "n": 3,
            "actions": ACTIONS,
        },
        "reward_range": [-1.0, 1.5],
    }

    def __init__(
        self,
        max_applications: int = 10,
        seed: int | None = None,
    ):
        self.max_applications = max_applications
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self._state: JobMarketState | None = None

    # ── Public API ──────────────────────────────

    def reset(self) -> JobMarketState:
        """Reset environment to a new episode. Returns initial state."""
        self._state = JobMarketState(
            skills_level=random.randint(1, 3),
            applications_left=self.max_applications,
            company_difficulty=random.randint(0, 2),
            shortlisted_count=0,
            upskill_count=0,
            wasted_count=0,
            episode_step=0,
            done=False,
        )
        return self._state

    def state(self) -> JobMarketState:
        """Return current state (read-only snapshot)."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state

    def step(self, action: int) -> tuple[JobMarketState, float, bool, dict[str, Any]]:
        """
        Execute action. Returns (next_state, reward, done, info).

        Args:
            action: int in {0, 1, 2}

        Returns:
            next_state : JobMarketState
            reward     : float
            done       : bool
            info       : dict with extra metadata
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode done. Call reset() to start a new one.")
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        s = self._state
        reward = 0.0
        info: dict[str, Any] = {
            "action_name": ACTIONS[action],
            "shortlisted": False,
        }

        # ── Action: apply small company ──
        if action == 0:
            s.applications_left -= 1
            p_success = min(0.20 + 0.10 * s.skills_level, 0.85)

            if random.random() < p_success:
                reward = 1.0
                s.shortlisted_count += 1
                info["shortlisted"]  = True
                info["message"]      = "✅ Shortlisted at small company!"
                info["rejection_reason"]      = None
                info["skill_recommendations"] = None
            else:
                reward = -1.0
                s.wasted_count += 1
                reason = get_rejection_reason(0, s.skills_level)
                recs   = get_skill_recommendations(0, s.skills_level)
                info["message"]               = f"❌ Rejected by small company."
                info["rejection_reason"]      = reason
                info["skill_recommendations"] = recs

        # ── Action: apply big company ──
        elif action == 1:
            s.applications_left -= 1
            difficulty_penalty = 0.08 * s.company_difficulty
            p_success = max(0.03, 0.05 + 0.10 * s.skills_level - difficulty_penalty)

            if random.random() < p_success:
                reward = 1.5
                s.shortlisted_count += 1
                info["shortlisted"]           = True
                info["message"]               = "🌟 Shortlisted at big company! Excellent!"
                info["rejection_reason"]      = None
                info["skill_recommendations"] = None
            else:
                reward = -1.0
                s.wasted_count += 1
                reason = get_rejection_reason(s.company_difficulty, s.skills_level)
                recs   = get_skill_recommendations(s.company_difficulty, s.skills_level)
                info["message"]               = f"❌ Rejected by big company."
                info["rejection_reason"]      = reason
                info["skill_recommendations"] = recs

        # ── Action: upskill ──
        elif action == 2:
            if s.skills_level < 5:
                s.skills_level += 1
                s.upskill_count += 1
                reward = 0.5
                # Show what new skill level unlocks
                curriculum = SKILL_CURRICULUM.get(s.company_difficulty, {})
                next_skills = curriculum.get("required_skills", [])
                unlocked = next_skills[min(s.skills_level - 1, len(next_skills) - 1)] if next_skills else "new techniques"
                info["message"]               = f"📚 Upskilled! ({s.skills_level}/5) — Now learning: {unlocked}"
                info["rejection_reason"]      = None
                info["skill_recommendations"] = get_skill_recommendations(s.company_difficulty, s.skills_level)
            else:
                reward = -0.2
                info["message"]               = "⚠️ Already at max skill level (5/5). This time was wasted."
                info["rejection_reason"]      = None
                info["skill_recommendations"] = {"priority": ["You are fully skilled — apply confidently!"],
                                                  "resources": [], "timeline": "Apply now"}

        # ── Episode termination ──
        s.episode_step += 1
        done = (s.applications_left <= 0) or (s.episode_step >= self.max_applications)
        s.done = done

        info["skills_level"] = s.skills_level
        info["applications_left"] = s.applications_left

        return s, reward, done, info

    # ── Utility ──────────────────────────────────

    def action_space_sample(self) -> int:
        """Sample a random valid action."""
        return random.randint(0, 2)

    def render(self) -> str:
        """Return a human-readable string of current state."""
        if self._state is None:
            return "Environment not initialized. Call reset()."
        s = self._state
        bars = "█" * s.skills_level + "░" * (5 - s.skills_level)
        apps = "●" * s.applications_left + "○" * (self.max_applications - s.applications_left)
        difficulty_labels = {0: "Startup 🏠", 1: "Mid-size 🏢", 2: "FAANG 🏦"}
        return (
            f"\n{'─'*40}\n"
            f"  Skills:      [{bars}] {s.skills_level}/5\n"
            f"  Apps left:   [{apps}] {s.applications_left}/{self.max_applications}\n"
            f"  Market:      {difficulty_labels[s.company_difficulty]}\n"
            f"  Shortlisted: {s.shortlisted_count}  |  Wasted: {s.wasted_count}  |  Upskills: {s.upskill_count}\n"
            f"  Step:        {s.episode_step}\n"
            f"{'─'*40}"
        )