"""
Interactive test script for Spotify Customer Support AI Agent.
Demonstrates end-to-end inference across diverse real-world customer tweets.
"""

import os
import sys

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import SupportAgent

def main():
    agent = SupportAgent()

    test_cases = [
        {
            "category": "Playback Issue (Auto-Handle)",
            "message": "offline downloaded songs won't play when I put my phone on airplane mode"
        },
        {
            "category": "App Bug / Crash (Auto-Handle)",
            "message": "Spotify desktop app crashes immediately on launch on macOS High Sierra"
        },
        {
            "category": "Catalogue / Licensing Query (Auto-Handle)",
            "message": "Why are all albums by Nena from 2002-2012 missing and greyed out?"
        },
        {
            "category": "Feature Request / Feedback (Auto-Handle)",
            "message": "Please add a custom clip sharing feature so I can share my favorite 10 seconds of a track"
        },
        {
            "category": "Billing Dispute (Escalate)",
            "message": "I was charged twice $9.99 for Spotify Premium this month! Refund my money!"
        },
        {
            "category": "Account Takeover / Security (Escalate)",
            "message": "Someone hacked into my account, changed the email address, and I cannot log in"
        },
        {
            "category": "Ambiguous Short Query (Escalate)",
            "message": "help it broke"
        }
    ]

    print("=" * 78)
    print("                SPOTIFY SUPPORT TRIAGE - LIVE INFERENCE TEST")
    print("=" * 78)

    for i, tc in enumerate(test_cases, 1):
        res = agent.process_message(tc["message"])
        print(f"\n[Test Case {i}] {tc['category']}")
        print(f"Customer Tweet : \"{tc['message']}\"")
        print(f"Intent         : {res['intent']} (Confidence: {res['intent_confidence']*100:.1f}%)")
        print(f"Decision       : {res['action']}")
        print(f"Reason         : {res['reason']}")
        if res.get("reply"):
            print(f"Drafted Reply  : \"{res['reply']}\"")
        top_case = res["retrieved_cases"][0] if res.get("retrieved_cases") else None
        if top_case:
            print(f"Retrieved Case : [{top_case['case_id']}] (similarity: {top_case['similarity']:.4f})")
            print(f"  Historical Q : \"{top_case['historical_issue']}\"")
            print(f"  Historical A : \"{top_case['historical_response']}\"")
        print("-" * 78)


if __name__ == "__main__":
    main()
