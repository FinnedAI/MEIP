PCT_DULL = 5
# This is the amount we want to dull the volatility of a day's change by
MAX_DAYS_DIFF = 1
# This is the maximum number of days we want to allow between the current day
# and the day we're predicting
SENT_PROB_DIST = list(range(-10, 11))
# The is the distribution of the sentiment probabilities, which is skewed to
# the right, as the average sentiment is roughly 0.2 (as determined by analysis)
SENT_PROBS = [
    0.0228,
    0.0359,
    0.0548,
    0.0808,
    0.1151,
    0.1587,
    0.2119,
    0.2473,
    0.3346,
    0.4207,
    0.5,
    0.4207,
    0.3346,
    0.2473,
    0.2119,
    0.1587,
    0.1151,
    0.0808,
    0.0548,
    0.0359,
    0.0228,
]
# This is the distribution of the sentiment probabilities, as determined from
# Z scores of the normal distribution.
SENT_FADE = 0.9
# This is the amount we want to fade the sentiment by each day to dampen
# the effects of the sentiment over time.
SENT_ABS_MIN = 0.01
# This is the minimum absolute value of the sentiment we want to allow
# before we reset the sentiment.
NEW_TICKER_FADE = 0.1
# This is the amount we want initialize as a penalty for predicting a new
# ticker, to avoid predicting the same ticker over and over.
TICKER_REPEATED_PENALTY = 0.9
# This is the amount we want to penalize a ticker for being predicted twice
# in a row.
TICKER_RESTORE = 1.1
# This is the amount we want to restore the penalty for a ticker that hasn't
# been predicted in a while.

