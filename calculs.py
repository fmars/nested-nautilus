import numpy as np
import pandas as pd


class Calculs:
    # --- MODIFICATION: Added 'reverse' flag to the constructor ---
    def __init__(self, n, ds, bezier_iterations=4, reverse=False):
        self.n = n # nombre de côté
        self.ds = ds # décalage
        self.ns = np.arange(3, self.n + 1) # les polygones de 3 côtés à n côtés
        self.bezier_iterations = bezier_iterations # Number of subdivisions for the Bezier curve
        self.reverse = reverse # Flag for reversing rotation

        self.get_info()
        
    def get_info(self):
        self.a = self.get_vertex() # longueur côtés

        self.in_radii = self.get_inner_circle_radius()
        self.out_radii = self.get_outer_circle_radius()
        self.inner_angles = self.get_inner_angles()
        self.outer_angles = self.get_outer_angles()
        self.d_rho, self.sd_rho, self.rots = self.get_rotations()
        
        # --- MODIFICATION: Removed self.aires ---
        # self.aires = self.get_aires() 
        self.pis = self.get_approx_pi() # approximation pi dans excel

        self.hyps, self.pgons, self.centres = self.get_polygons()

        self.dict_1 = self.get_dict_1()
        self.dict_2 = self.get_dict_2()
        self.df_1, self.df_2 = self.write_to_data_frame() # coordonnées

        self.spiral_x, self.spiral_y = self.get_spiral()
        self.pts = self.get_focus_pts()

        self.bezier_pts = self.get_courbe_bezier(self.bezier_iterations)

    def get_vertex(self):
        a = round(2 * 200 * np.sin(np.pi / self.n), 2)
        return round(a, 2)

    def get_inner_circle_radius(self):
        # rayon inscrit
        inner_radius = []
        for i in self.ns:
            r = round(0.5 * self.a * (1 / np.tan(np.pi / i)), 2)
            inner_radius.append(r)
        return inner_radius

    def get_outer_circle_radius(self):
        # rayon circonscrit
        outer_radius = []
        for i in self.ns:
            R = round(0.5 * self.a * (1 / np.sin(np.pi / i)), 1)
            outer_radius.append(R)
        return outer_radius

    def get_inner_angles(self):
        # rho
        inner_angles = []
        for i in self.ns:
            angle = round(180 * ((i - 2) / i) , 2)
            angle_rad = (angle / 180) * np.pi
            inner_angles.append(angle_rad)
        return inner_angles

    def get_outer_angles(self):
        # sigma
        outer_angles = []
        for i, j in zip(self.ns, self.inner_angles):
            angle = round(180 - (j / np.pi) * 180, 2)
            angle_rad = (angle / 180) * np.pi
            outer_angles.append(angle_rad)
        return outer_angles

    def get_rotations(self):
        delta_rho = np.abs(np.diff(self.inner_angles) / 2)
        delta_rho = np.append(delta_rho, 0)
        for n, dr in zip(self.ns, delta_rho):
            val = round(dr * (180 / np.pi), 2)

        somme_delta_rho = []
        for ind in range(len(delta_rho)):
            val = np.sum(delta_rho[ind:-1])
            somme_delta_rho.append(val)
            
        for n, sdr in zip(self.ns, somme_delta_rho):
            val = round(sdr * (180 / np.pi), 2)

        rots_fin = []
        for ind, val in enumerate(somme_delta_rho):
            if ind < len(somme_delta_rho) - 1:
                rot = val + (self.ds * np.sum(self.outer_angles[ind+1:]))
            else:
                rot = val
            
            # Apply reverse logic
            if self.reverse:
                rots_fin.append(-1 * rot)
            else:
                rots_fin.append(rot)

        for n, rot in zip(self.ns, rots_fin):
            val = round(rot * (180 / np.pi), 2)
        
        return delta_rho, somme_delta_rho, rots_fin

    # --- MODIFICATION: Area calculation function is no longer needed ---
    # def get_aires(self):
    #     # aire du plus grand polygone
    #     aires = []
    #     for i in self.ns:
    #         aire = 0.25 * i * self.a ** 2 * (1 / np.tan(np.pi / i))
    #         aires.append(round(aire, 2))
    #     return aires
    # --- END MODIFICATION ---

    # --- MODIFICATION: Updated Pi approximation method ---
    def get_approx_pi(self):
        approx_pis = []
        # Formula: pi ≈ n * sin(pi / n) where n is the number of sides
        for i in self.ns:
            pi_approx = i * np.sin(np.pi / i)
            approx_pis.append(round(pi_approx, 10)) # Increased precision
        return approx_pis
    # --- END MODIFICATION ---

    def get_polygons(self):
        polygons = []
        for ind in range(len(self.ns)):
            polygon = self.get_polygon(ind)
            polygons.append(polygon)

        distances, hyps = self.get_distances(polygons)
        polygons_fin = self.get_translations(polygons, distances)

        x_polygons, y_polygons = self.unpack_data(polygons_fin)
        return hyps, list([x_polygons, y_polygons]), distances

    def get_polygon(self, ind):
        points = []
        for i in range(self.ns[ind]):
            point_x = np.sin(self.outer_angles[ind] * i + self.rots[ind]) * self.out_radii[ind]
            point_y = np.cos(self.outer_angles[ind] * i + self.rots[ind]) * self.out_radii[ind]
            point = [point_x, point_y]
            points.append(point)
        points.append([points[0][0], points[0][1]])

        points_x = [round(i, 2) for i, j in points]
        points_y = [round(j, 2) for i, j in points]
        return list([np.array(points_x), np.array(points_y)])

    def get_distances(self, polygons):
        dist_x, dist_y = [[0, 0]], [[0, 0]]
        ind = len(polygons) - 1
        for i in polygons[::-1]:
            if ind >= 1:
                vx=i[0][self.ds % self.ns[ind]]
                vy=i[1][self.ds % self.ns[ind]]
                ux = polygons[ind-1][0][-1]
                uy = polygons[ind-1][1][-1]
                dy = uy - vy
                dx = ux - vx
                d = round(self.get_distance(dx, dy), 2)
                angle = np.arctan2(dy, dx)
                trans_x = round(d * np.cos(angle), 2)    # idem que dx
                trans_y = round(d * np.sin(angle), 2)    # idem que dy
                dist_x.append(trans_x)
                dist_y.append(trans_y)
            ind -= 1

        hyps, distances = [], []
        for i in range(len(dist_x)):
            val_x = round(np.sum(dist_x[::-1][i:-1]), 2)
            val_y = round(np.sum(dist_y[::-1][i:-1]), 2)
            hyp = round((val_x**2 + val_y**2) ** 0.5, 2)
            a = np.arctan2(val_y, val_x)
            distances.append([val_x, val_y])
            hyps.append(hyp)
        return distances, hyps

    def get_distance(self, dx, dy):
        d = np.sqrt(dx**2 + dy**2)
        return d

    def get_translations(self, polygons, distances):
        polygons_fin = []
        for i in range(len(self.ns)):
            coor_x = polygons[i][0] - distances[i][0]
            coor_y = polygons[i][1] - distances[i][1]
            polygons_fin.append([coor_x, coor_y])
        return polygons_fin

    def unpack_data(self, polygons_fin):
        xs, ys = [], []
        for coor, i in zip(polygons_fin, self.ns):
            xs.append(np.around(coor[0], 2))
            ys.append(np.around(coor[1], 2))
        return xs, ys

    def get_dict_1(self):
        # --- MODIFICATION: Removed "Aire" ---
        dict_1 = {
            "Rayon interne": self.in_radii,
            "Rayon externe": self.out_radii,
            "Angle interne": self.inner_angles,
            "Angle externe": self.outer_angles,
            "Delta rho": self.d_rho,
            "Somme delta rho": self.sd_rho,
            "Rotation finale": self.rots,
            "Distances": self.hyps,
            # "Aire": self.aires, # Removed
            "Pi approx.": self.pis
            }
        # --- END MODIFICATION ---
        return dict_1

    def get_dict_2(self):
        dict_2 = {}
        for val_x, val_y, i in zip(self.pgons[0], self.pgons[1], self.ns):
            delta = len(self.pgons[0][-1]) - len(val_x)
            dict_2[f"{i}-gon X"] = np.append(val_x, delta * [None])
            dict_2[f"{i}-gon Y"] = np.append(val_y, delta * [None])
        return dict_2

    def write_to_data_frame(self):
        df_1 = pd.DataFrame(self.dict_1, index=self.ns)
        df_2 = pd.DataFrame(self.dict_2)
        return df_1, df_2

    def get_spiral(self):
        spiral_x, spiral_y = [], []
        for i in self.ns:
            if i < self.ds:
                if self.ds % i == 0:
                    val_x = self.df_2[f"{i}-gon X"][i:i-2:-1]
                    val_y = self.df_2[f"{i}-gon Y"][i:i-2:-1]
                else:
                    val_x = self.df_2[f"{i}-gon X"][i:i-self.ds%i:-1]
                    val_y = self.df_2[f"{i}-gon Y"][i:i-self.ds%i:-1]
            else:
                val_x = self.df_2[f"{i}-gon X"][i:i-self.ds:-1]
                val_y = self.df_2[f"{i}-gon Y"][i:i-self.ds:-1]
            for x,y in zip(val_x, val_y):
                spiral_x.append(x)
                spiral_y.append(y)
        return spiral_x, spiral_y

    def get_focus_pts(self):
        pts = []
        for i in range(len(self.spiral_x)-2):
            systeme = ()
            for j in range(3):
                ind = i + j
                if j == 0:
                    point = (
                        (
                            (self.spiral_x[ind] + self.spiral_x[ind+1])/2,
                            (self.spiral_y[ind] + self.spiral_y[ind+1])/2,
                            )
                        ,)
                elif j == 1:
                    point = ((self.spiral_x[ind], self.spiral_y[ind]),)
                elif j == 2:
                    point = (
                        (
                            (self.spiral_x[ind] + self.spiral_x[ind-1])/2,
                            (self.spiral_y[ind] + self.spiral_y[ind-1])/2,
                            )
                        ,)
                systeme += (point)
            pts.append(systeme)
        return pts

    def get_courbe_bezier(self, subdivisions):
        if subdivisions < 0:
            return []
        nombre_de_subdivisions = subdivisions
        complets = []
        for courbe in self.pts:
            segments = [courbe]
            subdivision = 0

            while subdivision < nombre_de_subdivisions:
                nouveaux_segments = []
                for segment in segments:
                    milieu1_x = (segment[0][0] + segment[1][0]) / 2
                    milieu1_y = (segment[0][1] + segment[1][1]) / 2
                    milieu1 = (milieu1_x, milieu1_y)

                    milieu2_x = (segment[1][0] + segment[2][0]) / 2
                    milieu2_y = (segment[1][1] + segment[2][1]) / 2
                    milieu2 = (milieu2_x, milieu2_y)

                    milieu_final_x = (milieu1[0] + milieu2[0]) / 2
                    milieu_final_y = (milieu1[1] + milieu2[1]) / 2
                    milieu_final = (milieu_final_x, milieu_final_y)

                    nouveaux_segments.append((segment[0], milieu1, milieu_final))
                    nouveaux_segments.append((milieu_final, milieu2, segment[2]))

                segments = nouveaux_segments
                subdivision += 1
                if subdivision == nombre_de_subdivisions:
                    complets.append(segments)
        
        bezier_pts = []
        for seg in complets:
            for s in seg:
                bezier_pts.append(s[2])

        return bezier_pts
