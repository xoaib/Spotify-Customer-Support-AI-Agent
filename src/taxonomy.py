"""
Intent Taxonomy for Spotify Support Agent.

Derived empirically from 43,206 customer-brand conversation pairs for @SpotifyCares
in the Kaggle Customer Support on Twitter dataset (thoughtvector/customer-support-on-twitter).
"""

from typing import Dict, Any, List

INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "playback_issue": {
        "name": "playback_issue",
        "description": "Issues related to music/podcast playback: songs skipping, stopping unexpectedly, failing to buffer, stuttering, shuffle/repeat malfunctioning, or offline download playback errors.",
        "examples": [
            "My music keeps pausing after every single track on iOS",
            "offline playlists won't play when I have no internet connection",
            "shuffle button just repeats the same three songs",
            "songs randomly stop playing after 30 seconds"
        ],
        "default_action": "AUTO_HANDLE",
        "default_reason": "Standard playback troubleshooting: restart device, toggle offline mode, check connection, or check audio output settings.",
        "safe_to_auto_handle": True
    },
    "app_bug_crash": {
        "name": "app_bug_crash",
        "description": "Technical software defects: app crashing on startup, freezing, black screen, UI glitching, desktop minimize bug, or app failing to open.",
        "examples": [
            "Spotify desktop app crashes immediately upon opening on High Sierra",
            "App freezes on the launch screen every time I open it",
            "I uninstalled and reinstalled but the app still crashes",
            "Settings window won't save and keeps minimizing to taskbar"
        ],
        "default_action": "AUTO_HANDLE",
        "default_reason": "Safe to provide clean reinstall instructions, cache clearance, and OS version compatibility verification.",
        "safe_to_auto_handle": True
    },
    "content_availability": {
        "name": "content_availability",
        "description": "Questions regarding missing, greyed out, or unplayable songs, albums, or artists; queries about when music will return or why a track was removed.",
        "examples": [
            "Why did you guys remove Taylor Swift 1989 from the catalogue?",
            "An entire album is greyed out on my playlist and won't play",
            "Why is this track available in the UK but not in Canada?",
            "When are Nena's 2002-2012 albums coming back to Spotify?"
        ],
        "default_action": "AUTO_HANDLE",
        "default_reason": "Safe to explain music licensing agreements and rights holder availability, and direct to Spotify Community.",
        "safe_to_auto_handle": True
    },
    "subscription_billing": {
        "name": "subscription_billing",
        "description": "Inquiries regarding Premium plans, student discount verification, family plan billing, payment failures, double charges, or refund requests.",
        "examples": [
            "Why was I charged twice for Spotify Premium this month?",
            "How do I renew my student discount verification with SheerID?",
            "I cancelled my subscription last week but I was still charged $9.99",
            "Can I get a refund for an accidental renewal?"
        ],
        "default_action": "ESCALATE",
        "default_reason": "Billing disputes, unexpected charges, and refund requests involve private financial data and require human billing agent review.",
        "safe_to_auto_handle": False
    },
    "account_access_security": {
        "name": "account_access_security",
        "description": "Issues accessing the account: login errors, password reset failures, email address changed without permission, suspected compromised account, or Facebook login detachment.",
        "examples": [
            "Someone hacked my account and changed the email address, I can't log in",
            "I requested a password reset email three times and never received it",
            "Can't log in with my Facebook account anymore after the update",
            "My account is streaming songs from a different country that I didn't play"
        ],
        "default_action": "ESCALATE",
        "default_reason": "Account access and security issues pose privacy and takeover risks that require human verification.",
        "safe_to_auto_handle": False
    },
    "feature_request_feedback": {
        "name": "feature_request_feedback",
        "description": "Product suggestions, feedback on app design, user interface changes, ad experience feedback, or general praise/complaint without an actionable system outage.",
        "examples": [
            "Can we please get an option to block specific annoying ads?",
            "It would be awesome if Spotify had a custom audio clip sharing feature",
            "Please bring back the old lyrics UI, the new update is horrible",
            "Loving the new Release Radar playlist this week, great job!"
        ],
        "default_action": "AUTO_HANDLE",
        "default_reason": "Safe to thank the user, acknowledge feedback, and direct them to the Spotify Community Idea Exchange (community.spotify.com).",
        "safe_to_auto_handle": True
    }
}

VALID_INTENTS: List[str] = list(INTENT_DEFINITIONS.keys())


def is_valid_intent(intent: str) -> bool:
    """Check if an intent name exists in the taxonomy."""
    return intent in INTENT_DEFINITIONS


def get_intent_metadata(intent: str) -> Dict[str, Any]:
    """Retrieve full metadata for a given intent."""
    if not is_valid_intent(intent):
        raise ValueError(f"Unknown intent: '{intent}'. Valid intents are: {VALID_INTENTS}")
    return INTENT_DEFINITIONS[intent]


def get_taxonomy_prompt_text() -> str:
    """Format taxonomy for inclusion in LLM prompt."""
    lines = []
    for name, data in INTENT_DEFINITIONS.items():
        lines.append(f"- **{name}**: {data['description']}")
        lines.append(f"  *Examples*: {'; '.join(data['examples'][:2])}")
        lines.append(f"  *Standard Action*: {data['default_action']}")
    return "\n".join(lines)
