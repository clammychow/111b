import matplotlib.pyplot as plt
import random

class Polygons():
    def __init__(self, size, elongation, point, skew):
        if size < 0:
            raise ValueError("size cannot be negative")
        if elongation < 0:
            raise ValueError("elongation cannot be negative")
        if not (0 <= point <= 1):
            raise ValueError("point must be within [0, 1]")
        if not (-1 <= skew <= 1):
            raise ValueError("skew must be within [-1, 1]")
        self.size = size # width of base
        self.elongation = elongation # controls height relative to base
        self.point = point # controls distance between top vertices
        self.skew = skew # controls which direction top vertices lean
        self.params = [size, elongation, point, skew] # DOES NOT recompute parameters (used for generating initial lineup)
        # ensemble algorithm separately extracts parameters to create morph

    def verts(self): # centered on (0, 0)
        size = self.size
        elongation = self.elongation
        point = self.point
        skew = self.skew
        x_A = -(size/2) * (-point + 1) + (skew*size*point)/2
        y_A = (size*elongation)/2
        A = (x_A, y_A)
        x_B = (size/2) * (-point + 1) + (skew*size*point)/2
        y_B = (size*elongation)/2
        B = (x_B, y_B)
        x_C = size/2
        y_C = -(size*elongation)/2
        C = (x_C, y_C)
        x_D = -size/2
        y_D = -(size*elongation)/2
        D = (x_D, y_D)
        # for additional depth parameter, can copy all of these and just multiply depth by size
        # ex: E = (x_A, y_A, size*depth)
        return [A, B, C, D]
    
    def plot(self, color, ax):
        coords = self.verts()
        x_list = [p[0] for p in coords] + [coords[0][0]]
        y_list = [p[1] for p in coords] + [coords[0][1]]
        ax.plot(x_list, y_list, color=color)
        ax.scatter(x_list[:-1], y_list[:-1], color=color, s=30)
        return ax


#######################
# Algorithm Functions #
#######################

# Creates source shape parameters (modifiable) - used for generating lineup
def gen_source():
    shape = Polygons(
        size = random.uniform(6, 7), 
        elongation = random.uniform(0.5, 2), 
        point = random.uniform(0.2, 0.8), 
        skew = random.uniform(-1, 1)
        )
    return shape

# Generates several polygons that look similar to source shape
def gen_lineup(lineup_size, source):
    lineup = []
    for i in range(lineup_size):
        size, elongation, point, skew = source.params

        # generates noise around source parameters
        size = max(0, size*random.uniform(0.8, 1.2))
        elongation  = max(0, elongation*random.uniform(0.8, 1.2))
        point  = max(0, min(1,  point*random.uniform(0.6, 1.4)))
        skew = max(-1, min(1, skew*random.uniform(1, 1.5)))

        # creates polygon with new parameters
        poly = Polygons(size, elongation, point, skew)
        lineup.append(poly)
    return lineup

# Recomputes parameters from vertex locations - used for calculating morph parameters
def extract_params(poly):
    A, B, C, D = poly.verts()
    size = C[0] - D[0]
    height = A[1] - C[1]
    elongation = height/size
    top_width = B[0] - A[0]
    point = -(top_width/size) + 1
    if point == 0:
        skew = 0
    else:
        skew = (A[0] + (size/2)*(-point + 1)) / (size*point/2)
    return [size, elongation, point, skew]

# Computes averages of parameters and returns a morph of lineup shapes
def extract_ensemble(lineup):
    param_list = []
    for i in lineup:
        parameters = extract_params(i)
        param_list.append(parameters)
    avg_size = sum(i[0] for i in param_list) / len(lineup)
    avg_elongation = sum(i[1] for i in param_list) / len(lineup)
    avg_point = sum(i[2] for i in param_list) / len(lineup)
    avg_skew = sum(i[3] for i in param_list) / len(lineup)
    ensemble_poly = Polygons(avg_size, avg_elongation, avg_point, avg_skew)
    return ensemble_poly

# Full Algorithm Test
def ensemble_algorithm():
    test_shape = gen_source()
    test_lineup = gen_lineup(6, test_shape)
    test_morph = extract_ensemble(test_lineup)

    fig, axes = plt.subplots(1, 8, figsize=(15, 5))

    # Left Plot: Source
    test_shape.plot("red", ax=axes[0])
    axes[0].set_title("Source")
    axes[0].legend()

    # Middle Plot: Lineup
    for i, poly in enumerate(test_lineup):
        poly.plot("blue", ax=axes[i+1])
        axes[i+1].set_title(f"Lineup Polygon {i+1}")
        axes[i+1].legend()

    # Right Plot: Morph
    test_morph.plot("purple", ax=axes[7])
    axes[7].set_title("Ensemble Morph")
    axes[7].legend()

    for ax in axes:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

ensemble_algorithm()


#########################
# Recreating Face Study #
#########################

# Generate 6 lineup polygons from source shape A
# Generate 6 lineup polygons from source shape B (must have similar characteristics to A)
# Ask participant to study either lineup A or B in sequential or simultaneous format
# Ask participant to view 2 polygons and decide which polygon appeared in the studied lineup (2-AFC)
# 2-AFC Polygons could be:
#    > Ensemble morph of lineup A + Ensemble morph of lineup B
#    > Random polygon from lineup A + Random polygon from lineup B
#          ^^ used as baseline memory test for comparison against ensemble condition
