slack_token = "xoxb-3593791598916-3615419802016-6nDeUa8FR8Bfien3r5Cr1Dzk"
slack_channel = "#machine-learning"
slack_icon_emoji = ":see_no_evil:"
slack_user_name = "ML-Notifications"

import requests
import json


def post_message_to_slack(text, blocks=None):
    return requests.post(
        "https://slack.com/api/chat.postMessage",
        {
            "token": slack_token,
            "channel": slack_channel,
            "text": text,
            "icon_emoji": slack_icon_emoji,
            "username": slack_user_name,
            "blocks": json.dumps(blocks) if blocks else None,
        },
    ).json()


if __name__ == "__main__":
    post_message_to_slack("Test")
