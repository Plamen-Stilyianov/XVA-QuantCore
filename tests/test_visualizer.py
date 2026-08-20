import os
from risk_engine.visualizer import render_interactive_plotly_dashboard


def test_plotly_dashboard_html_generation():
    """
    Verifies that the visualizer module can generate and save
    the standalone interactive HTML dashboard file to disk.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # CORRECTED PATH LOOKUP: Point directly to the true output target directory
    target_html_path = os.path.join(project_root, "images", "xva_interactive_dashboard.html")

    # Remove any existing dashboard file to clear the test state
    if os.path.exists(target_html_path):
        os.remove(target_html_path)

    # Trigger the Plotly rendering loop
    render_interactive_plotly_dashboard()

    assert os.path.exists(target_html_path), "The visualizer module failed to export the HTML file to disk."
    assert os.path.getsize(target_html_path) > 0, "The exported interactive dashboard HTML file is empty."
