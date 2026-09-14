import sys
import unittest
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from similarity import preparar_matriz_similitud, recomendar  # noqa: E402


class SimilaritySmokeTest(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame(
            {
                "NAME": ["Referencia", "Cercano", "Lejano"],
                "TEAM": ["A", "B", "C"],
                "LEAGUE": ["L1", "L1", "L2"],
                "MAIN POSITION": ["Extremo", "Extremo", "Defensa central"],
                "AGE": [24, 23, 30],
                "MARKET VALUE": [20.0, 15.0, 5.0],
                "CONTRACT_OPPORTUNITY": [0, 1, 0],
                "GOALS_90_Z_LIGA": [0.0, 0.1, 2.0],
                "ASSISTS_90_Z_LIGA": [0.0, 0.1, -2.0],
                "POS_EXTREMO": [1.0, 1.0, 0.0],
                "POS_DEFENSA": [0.0, 0.0, 1.0],
            }
        )

    def test_nearest_player_is_returned_first(self):
        players, matrix = preparar_matriz_similitud(self.players)
        result = recomendar(players, matrix, "Referencia", n=1)

        self.assertIsNotNone(result)
        self.assertEqual(result.iloc[0]["NAME"], "Cercano")

    def test_budget_filter_is_respected(self):
        players, matrix = preparar_matriz_similitud(self.players)
        result = recomendar(
            players,
            matrix,
            "Referencia",
            n=3,
            presupuesto_max=10.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["NAME"].tolist(), ["Lejano"])


if __name__ == "__main__":
    unittest.main()
