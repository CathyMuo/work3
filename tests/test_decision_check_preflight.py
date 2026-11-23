import unittest

from work3.decision_check_preflight import probe


OPEN = "open cabinet 1"
GOOD = "take apple 1 from cabinet 1"
ALT = "take mug 1 from cabinet 1"
DROP = "put mug 1 in cabinet 1"
DONE = "put apple 1 in bowl 1"


def state(feedback, plan=(), legal=None, won=False):
    return {"feedback": feedback, "admissible_commands": plan if legal is None else legal,
            "facts": (feedback,), "won": won, "policy_commands": plan}


class ToyGame:
    def __init__(self, alternate_observations=False):
        self.resets = 0
        self.alternate_observations = alternate_observations

    def reset(self):
        self.resets += 1
        self.moves = []
        text = "You see cabinet 1."
        if self.alternate_observations and self.resets > 1:
            text += " The room changed."
        return state(text, [OPEN])

    def step(self, action):
        self.moves.append(action)
        result = {
            OPEN: state("The cabinet holds apple 1 and mug 1.", [GOOD], [GOOD, ALT]),
            GOOD: state("You take apple 1.", [DONE]),
            ALT: state("You take mug 1.", [DROP]),
            DROP: state("You put mug 1 down.", [GOOD]),
            DONE: state("You won.", won=True),
        }[action]
        return result, int(result["won"]), result["won"]


class DecisionCheckTests(unittest.TestCase):
    def test_actual_branch_cost_and_replay(self):
        game = ToyGame()
        result = probe(game)
        self.assertEqual(result["reason"], "qualified")
        self.assertEqual(result["preferred"], {"won": True, "steps_after_check": 2})
        self.assertEqual(result["alternative"], {"won": True, "steps_after_check": 4})
        self.assertEqual(game.moves, [OPEN, ALT, DROP, GOOD, DONE])

    def test_replay_drift_does_not_qualify(self):
        result = probe(ToyGame(alternate_observations=True))
        self.assertEqual(result, {"reason": "no_replayable_outcome_gap"})
