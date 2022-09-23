slack_token = "xoxb-3593791598916-3615419802016-POZAjANKfwk77iMYikOc8TgV"  # hardcoded token, channel will get deleted afterwards, if I forget, send me a message in the channel ;-)
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
    print(post_message_to_slack("Test"))
