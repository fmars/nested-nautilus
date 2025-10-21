# Nautilus Art Generator

Nautilus is an interactive web application for creating generative art based on the mathematical principles of nested polygons. Built with Python and Streamlit, it provides a user-friendly interface that doesn't require knowledge in coding.

## Features

* **Interactive Art Generation**: Create unique geometric art by adjusting parameters in a simple sidebar.
* **Customizable Parameters**: Control the number of polygons (`n`), the spiral offset (`ds`), color themes, rotation direction, and more.
* **Multiple Display Elements**: Choose to display various components of the figure, including polygons, inscribed/circumscribed circles, the main polygon spiral, and the resulting Bézier curve.
* **Dedicated Playgrounds**:
    * **Bézier Playground**: Isolate and experiment with the Bézier curve generated from the polygon spiral's control points.
    * **Logarithmic Spiral Explorer**: Plot and visualize classic logarithmic spirals by adjusting growth factors and turns.
* **Data Viewer**: Inspect the raw numerical data, including vertex coordinates and geometric properties, for each generated figure.
* **Generation History**: The app automatically saves your last 21 creations in a session, allowing you to easily revisit and compare them.
* **Built-in Gallery & Paper Viewer**: Includes a gallery of example images and a PDF viewer to display associated research or documentation.

## Setup and Installation

To run this application on your local machine, follow these steps:

### Prerequisites

* Python 3.8 or higher.

### Installation

1.  **Clone the repository:**
    ```
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The project relies on a few key libraries. Create a `requirements.txt` file with the following content:
    ```
    streamlit
    numpy
    pandas
    matplotlib
    ```
    Then, install them using pip:
    ```
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    Ensure your `main.py` and `calculs.py` files are in the same directory. Then run the following command in your terminal:
    ```
    streamlit run main.py
    ```
    The application should automatically open in a new tab in your web browser.

## File Structure

* `main.py`: The main Streamlit application script. It handles the user interface, session state, and calls the plotting logic.
* `calculation.py`: A module containing the `Calculation` class, which performs all the mathematical computations for generating polygon vertices, spiral points, and Bézier curve data.
* `paper.pdf` (Optional): A PDF document that can be displayed in the "Paper" tab.
* `gallery/` (Optional): A directory to store images for the "Gallery" tab.

## How to Use

1.  Open the application by running `streamlit run main.py`.
2.  Adjust the parameters in the sidebar on the left (e.g., Number of sides, Spiral offset, Color Theme).
3.  Click the **"Generate Art"** button to create a new figure.
4.  The generated image will appear in the **"Generator"** tab.
5.  Navigate through the other tabs (`History`, `Data`, `Bezier Playground`, etc.) to explore the different features of the application.

## Author

This project was created by **Maxime Chevillard**.
