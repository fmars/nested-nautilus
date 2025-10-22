import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from random import random, randint
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection, PatchCollection
from io import BytesIO
import base64 # To embed pdf
from collections import deque

from calculation import Calculation



class Plotter:
    """
    Plot the figure with matplotlib and create different tabs in the Streamlit app.
    """
    def __init__(self, info):
        self.info = info
        self.lbl_type = "sans-serif"
        
        # Set background and text color based on mode
        if self.info["mode"] == "dark":
            self.frame_color = "#0a0a0a"
            self.text_color = "#fafafa"
        else:
            self.frame_color = "#ffffff"
            self.text_color = "#0a0a0a"

    def plot_fig(self):
        """Generates and returns the Matplotlib figure based on user options."""
        # Pass the 'reverse' flag to Calculation
        try:
            self.data = Calculation(
                self.info["n"], 
                self.info["ds"], 
                reverse=self.info["reverse"]
            )
        except Exception as e:
            st.error(f"Error during calculations: {e}")
            # Return empty figure or handle error appropriately
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "Error in Calculation", ha='center', va='center')
            ax.axis('off')
            return fig, None # Return None for data if calculation fails

        # Initialize plot
        self.init_fig()
        self.init_flags()
        self.thickness_inv_sqrt()

        # Color theme
        color_map_name = self.info["theme"] 
        color_map_dict = {
            "Flag": "flag", "Spectrum": "jet", "Twilight": "twilight",
            "Binary": "binary", "Terrain": "terrain", "Ocean": "ocean",
            "Cividis": "cividis", "Clown": "RdGy"
        }
        
        num_polygons = max(1, self.data.n - 2) if hasattr(self.data, 'n') and self.data.n >=3 else 0
        num_bezier_pts = max(1, len(self.data.bezier_pts)) if hasattr(self.data, 'bezier_pts') else 0


        try: # Generate the colormaps
            if color_map_name == "Random 1":
                all_cmaps = plt.colormaps()
                # Filter out qualitative and reversed colormaps 
                quantitative_cmaps = { 
                    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
                }
                valid_cmaps = list(filter(
                    lambda c: not c.endswith('_r') and c not in quantitative_cmaps, all_cmaps
                ))
                rand_cmap_name = valid_cmaps[randint(0, len(valid_cmaps) - 1)]
                cmap = plt.colormaps.get_cmap(rand_cmap_name) 
            elif color_map_name == "Random 2":
                self.random_color_dict()
                cmap = LinearSegmentedColormap("random_cmap", self.cdict)
            else:
                cmap_name = color_map_dict.get(color_map_name, "jet") 
                cmap = plt.colormaps.get_cmap(cmap_name)

            colors = cmap(np.linspace(0.2, 1, num_polygons)) if num_polygons > 0 else []
            colors2 = cmap(np.linspace(0.2, 1, num_bezier_pts)) if num_bezier_pts > 0 else []

        except Exception as e:
             st.error(f"Error getting colormap '{color_map_name}': {e}")
             # Come back to default colors if failed
             default_cmap = plt.colormaps.get_cmap("jet")
             colors = default_cmap(np.linspace(0.2, 1, num_polygons)) if num_polygons > 0 else []
             colors2 = default_cmap(np.linspace(0.2, 1, num_bezier_pts)) if num_bezier_pts > 0 else []

        
        try: # Apply the colormaps to the figures
            # Check if data object and necessary attributes exist before plotting
            if self.data:
                 self.plot_polygon(colors)
                 self.plot_circle_inscribed(colors)
                 self.plot_circle_circumscribed(colors)
                 self.plot_circle_inscribed_circumscribed(colors)
                 self.plot_spiral()
                 self.plot_graph(colors)
                 self.plot_bezier(colors2)
                 self.plot_circle(colors)
                 self.plot_vertex(colors)
                 self.plot_segment()
                 self.plot_point()
        except AttributeError as e:
             st.error(f"Plotting error: Missing data - {e}. Check calculations.")
        except Exception as e:
             st.error(f"An unexpected error occurred during plotting: {e}")

        return self.fig, self.data

    def init_fig(self):
        mpl.rcParams["font.family"] = self.lbl_type
        self.fig, self.ax = plt.subplots(figsize=(8, 8)) 
        self.fig.patch.set_facecolor(self.frame_color)
        self.ax.set_facecolor(self.frame_color)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def init_flags(self):
        # Default to False if info keys are missing
        self.show_polygone = "Polygon" in self.info.get("display", [])
        self.show_ce_in = "Inscribed Circle" in self.info.get("display", [])
        self.show_ce_circon = "Circumscribed Circle" in self.info.get("display", [])
        self.show_spirale = "Polygon Spiral" in self.info.get("display", [])
        self.show_graphe = "Graph" in self.info.get("display", [])
        self.show_curve = "Bézier Curve" in self.info.get("display", [])
        self.show_cercle = "Circle" in self.info.get("display", [])
        self.show_vertex = "Vertex" in self.info.get("display", [])
        self.show_segment = "Segment" in self.info.get("display", [])
        self.show_point = "Point" in self.info.get("display", [])
        
        self.rempli_polygone = "Polygon" in self.info.get("fill", [])
        self.rempli_ce_in = "Inscribed Circle" in self.info.get("fill", [])
        self.rempli_ce_circon = "Circumscribed Circle" in self.info.get("fill", [])
        
        self.contour, self.couleur = self.info.get("contour", [False, "#000000"])


    def thickness_inv_sqrt(self):
        C = 4 * (3**0.5)
        # Ensure self.data exists and has attribute n
        n_val = getattr(self.data, 'n', 0)
        self.line_thickness = C / (n_val**0.5) if n_val > 0 else C


    def random_color_dict(self):    
        # For random colormap 2
        num_stops = randint(3, 12)
        positions = sorted([random() for _ in range(num_stops)])
        if 0.0 not in positions: positions.insert(0, 0.0)
        if 1.0 not in positions: positions.append(1.0)
        
        self.cdict = {"red": [], "green": [], "blue": []}
        for pos in positions:
            r, g, b = random(), random(), random()
            self.cdict["red"].append((pos, r, r))
            self.cdict["green"].append((pos, g, g))
            self.cdict["blue"].append((pos, b, b))


    def plot_polygon(self, colors):
        if not self.show_polygone or not hasattr(self.data, 'df_2') or self.data.df_2 is None: return
        # Ensure colors list matches ns length if needed
        safe_colors = colors[:len(self.data.ns)] if hasattr(self.data, 'ns') else []
        
        if self.rempli_polygone:
            for i, c in zip(self.data.ns, safe_colors):
                if f"{i}-gon X" in self.data.df_2 and f"{i}-gon Y" in self.data.df_2:
                    self.ax.fill(self.data.df_2[f"{i}-gon X"].dropna(), self.data.df_2[f"{i}-gon Y"].dropna(),
                                    color=c, alpha=1, zorder=self.data.n-i) # dropna added
        if self.contour:
            for i in self.data.ns:
                 if f"{i}-gon X" in self.data.df_2 and f"{i}-gon Y" in self.data.df_2:
                    self.ax.plot(self.data.df_2[f"{i}-gon X"].dropna(), self.data.df_2[f"{i}-gon Y"].dropna(),
                                    color=self.couleur, lw=self.line_thickness, alpha=1, zorder=self.data.n-i) # dropna added
        else:
            for i, c in zip(self.data.ns, safe_colors):
                 if f"{i}-gon X" in self.data.df_2 and f"{i}-gon Y" in self.data.df_2:
                    self.ax.plot(self.data.df_2[f"{i}-gon X"].dropna(), self.data.df_2[f"{i}-gon Y"].dropna(),
                                    color=c, lw=self.line_thickness, alpha=1, zorder=self.data.n-i) # dropna added
        self.ax.autoscale_view()

    def plot_circle_inscribed(self, colors):
         if not (self.show_ce_in and not self.show_ce_circon) or not hasattr(self.data, 'centres') or not hasattr(self.data, 'in_radii'): return
         transp = 0.6 if self.rempli_polygone and self.show_polygone else 1
         contour_colors = [self.couleur] * len(colors) if self.contour else colors
         min_len = min(len(self.data.centres), len(self.data.in_radii), len(colors), len(contour_colors))
         for pt, in_r, col, edge_col, i in zip(self.data.centres[:min_len], self.data.in_radii[:min_len], colors[:min_len], contour_colors[:min_len], range(min_len)):
             # Ensure pt is a valid coordinate pair and in_r is a number
             if isinstance(pt, (list, tuple)) and len(pt) == 2 and isinstance(in_r, (int, float)) and in_r >= 0:
                 c = plt.Circle((pt[0] * -1, pt[1] * -1), in_r, fill=self.rempli_ce_in, lw=self.line_thickness, edgecolor=edge_col, facecolor=col, alpha=transp, zorder=self.data.n - i)
                 self.ax.add_patch(c)
         self.ax.autoscale_view()


    def plot_circle_circumscribed(self, colors):
         if not (self.show_ce_circon and not self.show_ce_in) or not hasattr(self.data, 'centres') or not hasattr(self.data, 'out_radii'): return
         transp = 0.6 if self.rempli_polygone and self.show_polygone else 1
         contour_colors = [self.couleur] * len(colors) if self.contour else colors
         min_len = min(len(self.data.centres), len(self.data.out_radii), len(colors), len(contour_colors))
         for pt, out_r, col, edge_col, i in zip(self.data.centres[:min_len], self.data.out_radii[:min_len], colors[:min_len], contour_colors[:min_len], range(min_len)):
              if isinstance(pt, (list, tuple)) and len(pt) == 2 and isinstance(out_r, (int, float)) and out_r >= 0:
                 c = plt.Circle((pt[0] * -1, pt[1] * -1), out_r, fill=self.rempli_ce_circon, lw=self.line_thickness, edgecolor=edge_col, facecolor=col, alpha=transp, zorder=self.data.n - i)
                 self.ax.add_patch(c)
         self.ax.autoscale_view()


    def plot_circle_inscribed_circumscribed(self, colors):
         if not (self.show_ce_circon and self.show_ce_in) or not hasattr(self.data, 'centres') or not hasattr(self.data, 'in_radii') or not hasattr(self.data, 'out_radii'): return
         transp_inscrit, transp_circonscrit = (1, 0.6) if self.rempli_ce_in and self.rempli_ce_circon else (1, 1)
         if self.rempli_polygone and self.show_polygone:
             transp_inscrit = transp_circonscrit = 0.6
         contour_colors = [self.couleur] * len(colors) if self.contour else colors
         min_len = min(len(self.data.centres), len(self.data.in_radii), len(self.data.out_radii), len(colors), len(contour_colors))
         for pt, in_r, out_r, col, edge_col, i in zip(self.data.centres[:min_len], self.data.in_radii[:min_len], self.data.out_radii[:min_len], colors[:min_len], contour_colors[:min_len], range(min_len)):
             if isinstance(pt, (list, tuple)) and len(pt) == 2 and isinstance(in_r, (int, float)) and in_r >= 0 and isinstance(out_r, (int, float)) and out_r >= 0:
                 c_in = plt.Circle((pt[0] * -1, pt[1] * -1), in_r, fill=self.rempli_ce_in, lw=self.line_thickness, edgecolor=edge_col, facecolor=col, alpha=transp_inscrit, zorder=self.data.n - i)
                 c_out = plt.Circle((pt[0] * -1, pt[1] * -1), out_r, fill=self.rempli_ce_circon, lw=self.line_thickness, edgecolor=edge_col, facecolor=col, alpha=transp_circonscrit, zorder=self.data.n - i - 1)
                 self.ax.add_patch(c_in)
                 self.ax.add_patch(c_out)
         self.ax.autoscale_view()


    def plot_spiral(self):
        if not self.show_spirale: return
        if not hasattr(self.data, 'spiral_x') or not hasattr(self.data, 'spiral_y') or not self.data.spiral_x or not self.data.spiral_y: return
        to_check = [self.show_polygone, self.show_ce_in, self.show_ce_circon, self.show_curve,
                    self.show_cercle, self.show_vertex, self.show_segment, self.show_graphe, self.show_point]
        line_color = self.couleur if self.contour and not any(to_check) else self.text_color
        self.ax.plot(self.data.spiral_x, self.data.spiral_y, color=line_color, lw=self.line_thickness, zorder=self.data.n + 1)
        self.ax.autoscale_view()

    def plot_bezier(self, colors2):
         if not self.show_curve or not hasattr(self.data, 'bezier_pts') or not self.data.bezier_pts: return
         if len(self.data.bezier_pts) < 2: return
         
         # Ensure colors2 has enough elements or handle mismatch
         num_segments = len(self.data.bezier_pts) - 1
         safe_colors2 = colors2[:num_segments] if len(colors2) >= num_segments else ([colors2[0]] * num_segments if colors2 else ['blue'] * num_segments) # Fallback color
         
         line_colors = [self.couleur] * num_segments if self.contour else safe_colors2
         
         segments = []
         for i in range(num_segments):
             # Check if points are valid tuples/lists of len 2
             p1 = self.data.bezier_pts[i]
             p2 = self.data.bezier_pts[i+1]
             if (isinstance(p1, (tuple, list)) and len(p1) == 2 and
                 isinstance(p2, (tuple, list)) and len(p2) == 2):
                 segments.append([(p1[0], p1[1]), (p2[0], p2[1])])
             else:
                  # Skip invalid segment? Log warning?
                  print(f"Warning: Invalid points for Bezier segment at index {i}")
                  return # Stop plotting Bezier if data is bad

         if segments: # Only add collection if valid segments exist
             lc = LineCollection(segments, colors=line_colors, linewidths=self.line_thickness * 1.5, zorder=self.data.n + 1)
             self.ax.add_collection(lc)
             self.ax.autoscale_view()


    def plot_circle(self, colors):
         if not self.show_cercle or not hasattr(self.data, 'ns') or not hasattr(self.data, 'a') or not hasattr(self.data, 'df_2'): return
         patches, edge_colors = [], []
         base_colors = [self.couleur] * len(colors) if self.contour else colors
         min_len = min(len(self.data.ns), len(base_colors))
         radius = self.data.a / 2 if self.data.a > 0 else 0.1 # Ensure positive radius

         for i, col in zip(self.data.ns[:min_len], base_colors[:min_len]):
             if f"{i}-gon X" in self.data.df_2 and f"{i}-gon Y" in self.data.df_2:
                 xvals = self.data.df_2[f"{i}-gon X"].dropna() # Use dropna
                 yvals = self.data.df_2[f"{i}-gon Y"].dropna() # Use dropna
                 for ptsx, ptsy in zip(xvals, yvals):
                      # Check if ptsx, ptsy are numbers before creating Circle
                     if isinstance(ptsx, (int, float)) and isinstance(ptsy, (int, float)):
                         patches.append(plt.Circle((ptsx, ptsy), radius))
                         edge_colors.append(col)
         if patches: 
             pc = PatchCollection(patches, facecolor="none", edgecolor=edge_colors, linewidth=self.line_thickness, zorder=self.data.n + 1)
             self.ax.add_collection(pc)
             self.ax.autoscale_view()


    def plot_vertex(self, colors):
         if not self.show_vertex or not hasattr(self.data, 'ns') or not hasattr(self.data, 'centres') or not hasattr(self.data, 'df_2'): return
         line_colors = [self.couleur] * len(colors) if self.contour else colors
         line_segments, seg_colors = [], []
         min_len = min(len(self.data.ns), len(line_colors), len(self.data.centres))
         n_val = getattr(self.data, 'n', 1) # Get n for alpha calculation, default 1
         alpha_val = min(1, 1 / (n_val ** 0.5)) if n_val > 0 else 1

         for i, col in zip(self.data.ns[:min_len], line_colors[:min_len]):
             center_idx = i - 3
             if 0 <= center_idx < len(self.data.centres): # Check index validity
                 pt_center = self.data.centres[center_idx]
                 if isinstance(pt_center, (list, tuple)) and len(pt_center) == 2:
                     x_center, y_center = pt_center[0] * -1, pt_center[1] * -1
                     if f"{i}-gon X" in self.data.df_2 and f"{i}-gon Y" in self.data.df_2:
                         xvals = self.data.df_2[f"{i}-gon X"].dropna()
                         yvals = self.data.df_2[f"{i}-gon Y"].dropna()
                         for ptsx, ptsy in zip(xvals, yvals):
                             if isinstance(ptsx, (int, float)) and isinstance(ptsy, (int, float)):
                                 line_segments.append([(x_center, y_center), (ptsx, ptsy)])
                                 seg_colors.append(col)
         if line_segments: 
             lc = LineCollection(line_segments, colors=seg_colors, linewidths=self.line_thickness, alpha=alpha_val, zorder=self.data.n + 1)
             self.ax.add_collection(lc)
             self.ax.autoscale_view()


    def plot_graph(self, colors):
         if not (self.show_graphe and hasattr(self.data, 'ds') and self.data.ds != 0): return
         if not hasattr(self.data, 'ns') or not hasattr(self.data, 'df_2') or not hasattr(self.data, 'spiral_x') or not self.data.spiral_x: return

         line_colors = [self.couleur] * len(colors) if self.contour else colors
         line_segments, seg_colors = [], []
         min_len = min(len(self.data.ns), len(line_colors))
         n_val = getattr(self.data, 'n', 1)
         alpha_val = min(1, 1 / (n_val ** 0.5)) if n_val > 0 else 1

         for i, col, ind in zip(self.data.ns[:min_len], line_colors[:min_len], range(min_len)):
             if f"{i}-gon X" in self.data.df_2 and f"{i}-gon Y" in self.data.df_2:
                 xvals = self.data.df_2[f"{i}-gon X"].dropna()
                 yvals = self.data.df_2[f"{i}-gon Y"].dropna()
                 for ptsx, ptsy in zip(xvals, yvals):
                     spiral_index = ind * self.data.ds
                     if isinstance(ptsx, (int, float)) and isinstance(ptsy, (int, float)) and spiral_index < len(self.data.spiral_x):
                         spiral_pt_x = self.data.spiral_x[spiral_index]
                         spiral_pt_y = self.data.spiral_y[spiral_index]
                         # Ensure spiral points are also valid numbers
                         if isinstance(spiral_pt_x, (int, float)) and isinstance(spiral_pt_y, (int, float)):
                             line_segments.append([(spiral_pt_x, spiral_pt_y), (ptsx, ptsy)])
                             seg_colors.append(col)
         if line_segments:
             lc = LineCollection(line_segments, colors=seg_colors, linewidths=self.line_thickness, alpha=alpha_val, zorder=self.data.n + 1)
             self.ax.add_collection(lc)
             self.ax.autoscale_view()


    def plot_segment(self):
         if not self.show_segment: return
         if not hasattr(self.data, 'ds') or self.data.ds <= 0 or not hasattr(self.data, 'spiral_x') or len(self.data.spiral_x) < 2: return 
         
         to_check = [self.show_polygone, self.show_ce_in, self.show_ce_circon, self.show_curve,
                     self.show_cercle, self.show_vertex, self.show_spirale, self.show_graphe, self.show_point]
         line_color = self.couleur if self.contour and not any(to_check) else "red"
         
         nb = len(self.data.spiral_x)
         segments = []
         count = 0
         while count + 1 < nb: # Iterate up to the second to last point
             # Use current point and next point directly
             p1_x, p1_y = self.data.spiral_x[count], self.data.spiral_y[count]
             p2_x, p2_y = self.data.spiral_x[count+1], self.data.spiral_y[count+1]
             
             # Check if points are valid numbers
             if (isinstance(p1_x, (int, float)) and isinstance(p1_y, (int, float)) and
                 isinstance(p2_x, (int, float)) and isinstance(p2_y, (int, float))):
                  # Add segment every 'ds' steps
                 if count % self.data.ds == 0:
                      segments.append([(p1_x, p1_y), (p2_x, p2_y)])
             count += 1 # Increment count by 1 in each loop iteration

         if segments:
             lc = LineCollection(segments, colors=line_color, linewidth=self.line_thickness, linestyle="-", zorder=self.data.n + 2 if hasattr(self.data, 'n') else 2)
             self.ax.add_collection(lc)
             self.ax.autoscale_view()


    def plot_point(self):
         if not self.show_point: return
         if not hasattr(self.data, 'spiral_x') or not hasattr(self.data, 'spiral_y') or not self.data.spiral_x or not self.data.spiral_y: return 
         
         # Filter out potential None or non-numeric values
         valid_x = [x for x in self.data.spiral_x if isinstance(x, (int, float))]
         valid_y = [y for y in self.data.spiral_y if isinstance(y, (int, float))]
         
         # Ensure x and y have the same length after filtering
         min_len = min(len(valid_x), len(valid_y))
         valid_x = valid_x[:min_len]
         valid_y = valid_y[:min_len]

         if not valid_x: return # No valid points to plot
         
         n_val = getattr(self.data, 'n', 1)
         point_size = max(10, 100 / (n_val**0.5)) if n_val > 0 else 100
         
         self.ax.scatter(valid_x, valid_y, s=point_size, c="red", marker=".", zorder=self.data.n + 2 if hasattr(self.data, 'n') else 2)
         self.ax.autoscale_view()



# Streamlit app main function
def main():
    st.set_page_config(
        page_title="Nautilus",
        page_icon="./gallery/icon.png",     
        initial_sidebar_state="expanded"
    )


    st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state for history
    st.session_state.setdefault('figure_history', deque(maxlen=21))

    # Sidebar for choosing the parameters
    with st.sidebar:
        st.title("Nautilus")
        st.caption("Made by *Maxime Chevillard*") 
        st.header("Parameters") 

        n_sides = st.slider("Number of sides (n)", min_value=3, max_value=100, value=20, step=1)
        ds_offset = st.slider("Spiral offset (ds)", min_value=0, max_value=3, value=1, step=1)

        theme_options = ["Random 1", "Random 2", "Spectrum", "Twilight", "Binary", "Flag", "Terrain", "Ocean", "Cividis", "Clown"]
        theme_choice = st.selectbox("Color Theme", theme_options)

        display_options = ["Polygon", "Inscribed Circle", "Circumscribed Circle", "Polygon Spiral", "Graph", "Bézier Curve", "Circle", "Vertex", "Segment", "Point"]
        display_final_choices = st.multiselect("Elements to Display", display_options, default=["Polygon", "Polygon Spiral"])

        fillable_options = [opt for opt in ["Polygon", "Inscribed Circle", "Circumscribed Circle"] if opt in display_final_choices]
        remplissage_final_choices = []
        if fillable_options:
            remplissage_final_choices = st.multiselect("Elements to Fill", fillable_options, default=fillable_options)
        
        st.markdown("---")
        contour_choice = st.checkbox("Add Custom Contour")
        reverse_choice = st.checkbox("Reverse Rotation", value=False)
        
        contour_color = "#000000"
        if contour_choice:
            contour_color = st.color_picker("Contour Color", '#FF5733')
        
        mode_choice = st.radio("Background Mode", ["light", "dark"], index=0)

        st.markdown("---")
        generate_button = st.button("Generate", width='stretch')

    tabs = st.tabs([
        "Generator", "History", "Data", "Gallery", "Bézier Playground", "Paper", "Help", "Acknoledgement"
    ])


    with tabs[0]: # Generator
        if generate_button:
            info_dict = {
                "display": display_final_choices, 
                "theme": theme_choice, 
                "fill": remplissage_final_choices, 
                "contour": [contour_choice, contour_color], 
                "n": n_sides, 
                "ds": ds_offset, 
                "mode": mode_choice,
                "reverse": reverse_choice
            }
            with st.spinner('Generating your masterpiece...'):
                try:
                    plotter_instance = Plotter(info_dict)
                    fig, data = plotter_instance.plot_fig()
                    # Check if fig is valid before storing
                    if fig is not None:
                        st.session_state.fig = fig
                        st.session_state.data = data # Store data only if fig is valid
                        st.session_state.n_sides = n_sides
                        st.session_state.ds_offset = ds_offset
                        # --- MODIFICATION: Add figure to history ---
                        # Use fig.copy() if available, otherwise just append fig
                        try:
                            st.session_state.figure_history.appendleft(fig.copy()) # Add newest to the left
                        except AttributeError: # Older matplotlib might not have copy
                            st.session_state.figure_history.appendleft(fig)
                        # --- END MODIFICATION ---

                    else:
                        st.error("Failed to generate figure.")
                        # Clear previous figure/data if generation failed
                        if 'fig' in st.session_state: del st.session_state.fig
                        if 'data' in st.session_state: del st.session_state.data
                except Exception as e:
                    st.error(f"Error during art generation: {e}")
                    # Clear previous figure/data on error
                    if 'fig' in st.session_state: del st.session_state.fig
                    if 'data' in st.session_state: del st.session_state.data


        if 'fig' in st.session_state and st.session_state.fig is not None:
            # Display the generated figure
            try:
                st.pyplot(st.session_state.fig, width='stretch')
            except Exception as e:
                st.error(f"Error displaying plot: {e}")
        elif not generate_button: # Only show info if not attempting generation
            st.info("Adjust the settings in the sidebar and click 'Generate' to create your image.")

    with tabs[1]: # History
        if 'figure_history' in st.session_state and st.session_state.figure_history:
            history_list = list(st.session_state.figure_history) # Convert deque to list for easier indexing
            num_figures = len(history_list)
            num_rows = (num_figures + 2) // 3 # Calculate rows needed

            for i in range(num_rows):
                cols = st.columns(3)
                for j in range(3):
                    fig_index = i * 3 + j
                    if fig_index < num_figures:
                        with cols[j]:
                            try:
                                st.pyplot(history_list[fig_index], width="stretch")
                            except Exception as e:
                                st.error(f"Error displaying history figure {fig_index+1}: {e}")
                    else:
                        # Add empty space in the last row if needed
                        with cols[j]:
                            st.empty()
        else:
            st.info("No figures generated yet in this session.")

    # Data tab
    with tabs[2]: # Data

        if 'data' in st.session_state and st.session_state.data is not None:
            st.markdown("### General Polygon Information")
            # Check if df_1 exists and is a DataFrame
            if hasattr(st.session_state.data, 'df_1') and isinstance(st.session_state.data.df_1, pd.DataFrame):
                df1_to_display = st.session_state.data.df_1.copy() # Work on a copy
                numeric_cols_df1 = df1_to_display.select_dtypes(include=np.number).columns
                st.dataframe(df1_to_display)
            else:
                st.warning("General polygon information data (df_1) is not available or invalid.")

            st.markdown("### Polygon Vertex Coordinates")
            # Check if df_2 exists and is a DataFrame
            if hasattr(st.session_state.data, 'df_2') and isinstance(st.session_state.data.df_2, pd.DataFrame):
                df2_to_display = st.session_state.data.df_2.copy() # Work on a copy
                numeric_cols_df2 = df2_to_display.select_dtypes(include=np.number).columns
                st.dataframe(df2_to_display)
            else:
                st.warning("Polygon vertex coordinate data (df_2) is not available or invalid.")
        else:
            st.info("Generate an image in the 'Generator' tab to see its data here.")

    
    # Gallery Tab 
    with tabs[3]: 
        images_path = [f"./gallery/image{i}.jpg" for i in range(1, 22)]

        if len(images_path) == 0:
            st.info("The gallery is empty.")
        else:
            for i in range(0, len(images_path), 3): # rows
                cols = st.columns(3)
                for j in range(3): # 3 columns
                    with cols[j]:
                        try:
                            st.image(images_path[i + j], width=200)
                        except Exception:
                            st.warning(f"Could not load image{i+j}.")

    # Bezier Playground tab
    with tabs[4]:
        
        # Controls above the plot
        n_poly = st.slider("Number of sides (n)", 3, 20, 15, key="n_bezier")
        ds_bezier = st.slider("Spiral Offset (ds)", 1, 2, 1, key="ds_bezier")
        iter_bezier = st.slider("Bezier Subdivisions", 0, 8, 4, key="iter_bezier", help="Number of recursive subdivisions (0 = control points only).")
        show_controls = st.checkbox("Show Control Points", value=True)

        try:
            data_bezier = Calculation(n=n_poly, ds=ds_bezier, bezier_iterations=iter_bezier)
            
            # Calculate plot limits 
            plot_lims = (-100, 100) 
            x_center, y_center = 0, 0
            # Check if necessary attributes exist before calculating limits
            if hasattr(data_bezier, 'ns') and len(data_bezier.ns) > 0 and hasattr(data_bezier, 'df_2') and not data_bezier.df_2.empty:
                all_x_list = [data_bezier.df_2[f"{i}-gon X"].dropna() for i in data_bezier.ns if f"{i}-gon X" in data_bezier.df_2]
                all_y_list = [data_bezier.df_2[f"{i}-gon Y"].dropna() for i in data_bezier.ns if f"{i}-gon Y" in data_bezier.df_2]
                
                if all_x_list and all_y_list: 
                    all_x = np.concatenate(all_x_list)
                    all_y = np.concatenate(all_y_list)
                    if all_x.size > 0 and all_y.size > 0: 
                        x_min, x_max = all_x.min(), all_x.max()
                        y_min, y_max = all_y.min(), all_y.max()
                        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
                        range_x = x_max - x_min if x_max > x_min else 1
                        range_y = y_max - y_min if y_max > y_min else 1
                        max_range = max(range_x, range_y) * 1.1 
                        plot_lims = (x_center - max_range / 2, x_center + max_range / 2)

            fig_bezier, ax_bezier = plt.subplots(figsize=(8, 8)) 
            
            # Check attributes before plot
            if hasattr(data_bezier, 'spiral_x') and hasattr(data_bezier, 'spiral_y') and data_bezier.spiral_x and data_bezier.spiral_y:
                ax_bezier.plot(data_bezier.spiral_x, data_bezier.spiral_y, color='#0077b6', lw=1.5, ls='--', label="Polygon Spiral")
            
            if hasattr(data_bezier, 'bezier_pts') and data_bezier.bezier_pts:
                # Ensure bezier_pts contains valid points before zipping
                valid_bezier_pts = [p for p in data_bezier.bezier_pts if isinstance(p, (tuple, list)) and len(p) == 2]
                if len(valid_bezier_pts) > 1:
                    bz_x, bz_y = zip(*valid_bezier_pts)
                    ax_bezier.plot(bz_x, bz_y, color='#d00000', lw=2, label="Bezier Curve")

            if show_controls and hasattr(data_bezier, 'pts') and data_bezier.pts:
                # Filter valid point before processing
                valid_pts = [p for p in data_bezier.pts if isinstance(p, (tuple, list)) and len(p) == 3 and 
                                    all(isinstance(pt, (tuple, list)) and len(pt)==2 for pt in p)]
                if valid_pts:
                    p1s_x = [p[1][0] for p in valid_pts]
                    p1s_y = [p[1][1] for p in valid_pts]
                    ax_bezier.scatter(p1s_x, p1s_y, c='red', s=20, label="Control Points (P1)", zorder=5)
                    
                    p0s_x = [p[0][0] for p in valid_pts]
                    p0s_y = [p[0][1] for p in valid_pts]
                    p2s_x = [p[2][0] for p in valid_pts]
                    p2s_y = [p[2][1] for p in valid_pts]
                    ax_bezier.scatter(p0s_x, p0s_y, c='gray', s=10, label="Anchor Points (P0, P2)", zorder=4)
                    ax_bezier.scatter(p2s_x, p2s_y, c='gray', s=10, zorder=4)
                    
                    for p in valid_pts:
                        ax_bezier.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color='gray', lw=0.5, ls=':', zorder=3)
                        ax_bezier.plot([p[1][0], p[2][0]], [p[1][1], p[2][1]], color='gray', lw=0.5, ls=':', zorder=3)

            fig_bezier.patch.set_facecolor("white")
            ax_bezier.set_facecolor("white")
            ax_bezier.set_xlim(plot_lims)
            ax_bezier.set_ylim(plot_lims)
            # Center axes only if center is reasonably calculated
            if x_center is not None and y_center is not None:
                try:
                    ax_bezier.spines['left'].set_position(('data', x_center))
                    ax_bezier.spines['bottom'].set_position(('data', y_center))
                except ValueError: # Handle cases where center might be outside limits
                    ax_bezier.spines['left'].set_position('zero')
                    ax_bezier.spines['bottom'].set_position('zero')

            ax_bezier.spines['right'].set_color('none')
            ax_bezier.spines['top'].set_color('none')
            # Check limits before plotting arrows
            if plot_lims[0] is not None and plot_lims[1] is not None and y_center is not None and x_center is not None:
                ax_bezier.plot(plot_lims[1], y_center, ">", color='black', clip_on=False)
                ax_bezier.plot(x_center, plot_lims[1], "^", color='black', clip_on=False)
            ax_bezier.legend()
            
            st.pyplot(fig_bezier) 

        except Exception as e:
            st.error(f"Error in Bezier Playground: {e}")


    # Paper Tab
    with tabs[5]: 
        try:
            with open("paper.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf" style="border: none;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("`nautilus-polygon.pdf` not found.")
        except Exception as e:
            st.error(f"Error loading PDF: {e}")


    # Help Tab
    with tabs[6]:  # Help
        st.markdown("### Generator")
        st.markdown("This section displays the figures generated using the parameters set in the sidebar.")

        col1, col2 = st.columns(2)
        with col1:
            st.info("**Number of sides (n)**")
            st.markdown("Defines the maximum number of sides for the largest polygon. A higher `n` value creates a denser, more circular figure.")
        with col2:
            st.info("**Spiral offset (ds)**")
            st.markdown("Controls how the vertices of adjacent polygons connect. `ds = 1` creates a smooth spiral, while `ds = 2` or `ds = 3` produces more complex patterns.")

        col3, col4 = st.columns(2)
        with col3:
            st.info("**Color Theme**")
            st.markdown("`Random 1` selects a random predefined palette from the Matplotlib library. `Random 2` generates random colors to create color gradients.")
        with col4:
            st.info("**Elements to Display**")
            st.markdown("Displays multiple figures based on the same principles—nested polygons with an offset between them. Multiple elements can be selected at once.")

        col5, col6 = st.columns(2)
        with col5:
            st.info("**Elements to Fill**")
            st.markdown("When `Polygon`, `Inscribed Circle`, or `Circumscribed Circle` are selected under `Elements to Display`, you can choose whether to fill these shapes.")
        with col6:
            st.info("**Download Image**")
            st.markdown("After generating a figure, the user can download the image directly through their browser.")

        st.markdown("---")
        st.markdown("### History")
        st.markdown("This section displays the 21 most recent figures generated, with the latest shown first.")

        st.markdown("---")
        st.markdown("### Data")
        st.markdown("For a detailed explanation of the parameters in the table, please refer to the accompanying paper.")
        st.markdown("In the coordinate table, there is always one more line than the number of polygon sides because it also includes the order in which the polygon segments are drawn. Take the triangle (3-gon) for example, we start from the first coordinate point, draw to the second, then the third, and finally return to the first point. Therefore, the first and last coordinates are the same.")

        st.markdown("---")
        st.markdown("### Gallery")
        st.markdown("The gallery includes images generated directly using this application, as well as some variations created by adjusting certain parameters to explore different visual results.")

        st.markdown("---")
        st.markdown("### Bézier Playground")
        st.markdown("In this tab, users can experiment with the internal workings of the `Bézier curve` feature used in `Elements to Display`. While the main program uses a fixed number of iterations, this section allows you to explore how the Bézier algorithm behaves with different iteration counts.")

        st.markdown("---")
        st.markdown("### Paper")
        st.markdown("This short paper explains how the coordinates of the polygons were calculated for different offsets, and draws an interesting resemblance with the nautilus shell. It explores various spiral types and compares them to those generated with Bézier curves — which, interestingly, are quite close to logarithmic spirals with a growth coefficient near √φ ≈ 1.272.")

        st.markdown("---")
        st.markdown("### Settings")
        st.markdown("Users can adjust the settings from the three-dot menu at the top right corner. Options include changing the background theme and enabling wide mode, depending on the device being used.")

        

    with tabs[7]:
        st.markdown("### Acknowledgements")

        st.markdown(
            "This project was inspired in part by the Swiss artist Max Bill and his piece *“Fifteen Variations on a Single Theme”* (1938), "
            "as well as Jean-Pierre Hébert and his series *“One Hundred Views of a Metagon.”* "
            "I hope this project honors their creativity and helps more people discover their amazing work."
        )

        # center the image
        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:
            st.image(
                "./gallery/maxbill.jpg",
                caption="Max Bill: *Fifteen Variations on a Single Theme* (1938). Photo by TenerifeTenerife, licensed under CC.",
                width="stretch"
            )
        st.markdown(
            "If you’d like, you can even try to recreate similar figures using this program. "
            "Some of the patterns you’ll get might look familiar. But others, especially those using the *reverse* option or different settings, "
            "are unique to this software and completely original."
        )

        st.markdown(
            "The website is built entirely in Python, and the full source code is available on my GitHub. "
            "It’s meant to be both a learning resource and a fun starting point for artists, designers, or anyone curious about generative art. "
            "I also have a version that runs with custom Tkinter instead of Streamlit, which I can share if there’s interest."
        )

        st.markdown("Have a great day!")
        st.markdown("*— Maxime Chevillard, Paris*")
        st.info("Contact: maxime.chevillardsnc@gmail.com")

        st.markdown("---")
        st.markdown(
            """
            ```
            MIT License © 2025 Maxime Chevillard

            Permission is granted to use, copy, modify, and distribute this software,
            provided the copyright notice is included.
            The software is provided "as is" without warranty.
            ```
            """
        )


if __name__ == "__main__":
    main()
