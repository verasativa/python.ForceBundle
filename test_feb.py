import unittest, csv, ast
import ForcedirectedEdgeBundling as feb

class TestUM(unittest.TestCase):
    def setUp(self):
        # Load raw test data
        with open('test_data/airlines.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            edges = feb.get_empty_edge_list()

            for row in csvreader:
                source = feb.Point(float(row[0]), float(row[1]))
                target = feb.Point(float(row[2]), float(row[3]))
                edge = feb.Edge(source, target)
                edges.append(edge)

        # Apply main algorithm
        bundled_lines = feb.forcebundle(edges)

        # Convert output to native python object (where we can perform logical comparison)
        self.output_list = []
        for line in bundled_lines:
            points = []
            for point in line:
                points.append((point.x, point.y))
            self.output_list.append(points)

        # Load processed test data
        with open('test_data/airlines_bundled.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            self.tested_list = []

            for row in csvreader:
                line = [ast.literal_eval(point) for point in row]
                self.tested_list.append(line)

    # End to end test
    def test_end2end(self):
        self.assertEqual(self.output_list, self.tested_list)

if __name__ == '__main__':
    unittest.main()