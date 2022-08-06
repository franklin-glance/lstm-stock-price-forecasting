API_KEY = 'SBIABVRDNL5921LU'
import time


def get_api_key():
    return API_KEY


class env:
    def __init__(self):
        self.api_calls = 0
        self.last_api_call = 0

    def reset_api(self):
        self.api_calls = 0
        self.last_api_call = 0

    def make_api_request(self):
        self.api_calls += 1
        if self.api_calls % 75 == 0:
            amt = 75 - (time.time() - self.last_api_call)
            if amt > 0:
                print(f'sleeping for {amt}s to avoid api rate limit')
                time.sleep(amt)
                self.last_api_call = time.time()