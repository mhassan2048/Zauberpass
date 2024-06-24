import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from mplsoccer import Pitch
import matplotlib.image as mpimg
import io
import base64
import tempfile
import matplotlib.font_manager as fm

# Add custom font
font_path = 'DIN-Condensed-Bold.ttf'
font_prop = fm.FontProperties(fname=font_path)
font_prop_large = fm.FontProperties(fname=font_path, size=48, weight='bold')
font_prop_medium = fm.FontProperties(fname=font_path, size=24, weight='bold')
font_prop_small = fm.FontProperties(fname=font_path, size=20, weight='bold')


# Initialize global variables for text objects
minute_text_obj = None
receiver_text_obj = None

# Custom functions from the original code
def is_long_pass(x_start, x_end):
    dist_x = np.abs(x_end - x_start)
    return ((x_start < 50 and x_end < 50 and x_end > x_start and dist_x >= 30) or 
            (x_start < 50 and x_end > 50 and dist_x >= 15) or 
            (x_start > 50 and x_end > 50 and x_end > x_start and dist_x >= 10))

def draw_pass(ax, row, pitch, comp_clr, regular_clr, failed_clr, key_pass_clr):
    if 'CornerTaken' not in row['qualifiers']:
        x_start, y_start, x_end, y_end = row['x'], row['y'], row['end_x'], row['end_y']
        pass_color = regular_clr if row['outcome_type'] == 'Successful' else failed_clr
        pass_width = 5 if row['outcome_type'] == 'Successful' else 4
        if 'KeyPass' in row['qualifiers']:
            pass_color = key_pass_clr
            pass_width = 6
            zorder=5
        if is_long_pass(x_start, x_end):
            pitch.lines(xstart=x_start, ystart=y_start, xend=x_end, yend=y_end, 
                        color=comp_clr, lw=5, zorder=4, transparent=True, 
                        alpha_start=1, alpha_end=0.01, ax=ax)
        else:
            pitch.lines(xstart=x_start, ystart=y_start, xend=x_end, yend=y_end, 
                        color=pass_color, lw=pass_width, zorder=3, transparent=True, 
                        alpha_start=0.75, alpha_end=0.01, ax=ax)

def draw_heatmap(ax, pass_events, pitch, clr_map):
    pass_end_locations = pass_events[['end_x', 'end_y']].to_numpy()
    bin_statistic = pitch.bin_statistic_positional(pass_end_locations[:, 0], pass_end_locations[:, 1], statistic='count', positional='full', normalize=True)
    pitch.heatmap_positional(bin_statistic, ax=ax, cmap=clr_map, edgecolors='lightgrey', linewidth=5, zorder=1, alpha=.03)

def update(frame, ax, pitch, pass_events_sorted, comp_clr, regular_clr, failed_clr, key_pass_clr, clr_map):
    global minute_text_obj, receiver_text_obj

    if frame < len(pass_events_sorted):
        row = pass_events_sorted.iloc[frame]
        draw_pass(ax, row, pitch, comp_clr, regular_clr, failed_clr, key_pass_clr)

        minute = row['minute']
        receiver = 'N/A'
        if row['outcome_type'] == 'Successful':
            try:
                next_row = df.iloc[df.index.get_loc(row.name) + 1]
                if next_row['team'] == row['team']:
                    receiver = next_row['player']
                else:
                    receiver = 'Opponent Action'
            except (IndexError, ValueError):
                receiver = 'N/A'

        if minute_text_obj:
            minute_text_obj.remove()
        if receiver_text_obj:
            receiver_text_obj.remove()

        font_properties_bottom = {'family': 'DIN Condensed', 'weight': 'bold', 'size': 20}
        minute_text_obj = plt.figtext(0.95, 0.08, f"Minute: {minute}", fontdict=font_properties_bottom, color='w', ha='right')
        receiver_text_obj = plt.figtext(0.95, 0.06, f"Receiver: {receiver}", fontdict=font_properties_bottom, color='#2af5bf', ha='right')

        draw_heatmap(ax, pass_events_sorted.iloc[:frame + 1], pitch, clr_map)

    return []

def load_data():
    # Update the URL to your Google Drive direct download link
    url = "https://drive.google.com/uc?export=download&id=1iKLZKodLUMa9akCyCUhgS15bur4Hxu4t"
    df = pd.read_csv(url)
    return df

# Streamlit app
st.title("Zauberpass by The Real Deal - Euro 2024 Edition")
st.sidebar.image("zplogo.png", use_column_width=True)  # Add your logo here

# Load data
df = load_data()

# Extract unique teams
teams = sorted(df['team'].unique())
selected_team = st.sidebar.selectbox("Select Team", teams, index=0)

# Filter matches based on the selected team
filtered_df_team = df[df['team'] == selected_team]
matches = sorted(filtered_df_team['game'].unique())
selected_match = st.sidebar.selectbox("Select Match", matches, index=0)

# Filter players based on the selected team and game
filtered_df_game = filtered_df_team[filtered_df_team['game'] == selected_match]
players = sorted(filtered_df_game['player'].dropna().unique())
selected_player = st.sidebar.selectbox("Select Player", players, index=0)

if st.button("Show Passmap"):
    # Filter data based on selections
    match_df = filtered_df_game
    player_passes = match_df[(match_df['player'] == selected_player) & (match_df['type'] == 'Pass')]
    pass_events_sorted = player_passes.sort_values(by=['minute', 'second'])

    first_game_info = pass_events_sorted.iloc[0]['game'] if len(pass_events_sorted) > 0 else 'Game Information Not Available'

    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='#6F0049')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('#6F0049')
    ax.patch.set_facecolor('#6F0049')
    plt.gca().invert_yaxis()
    
    # Load your image
    image_path = 'brand4.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.8, 0.84, 0.15, 0.15])  # Example: [left, bottom, width, height]
    
    # Display the image in the new axes
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    # Create an axis for the colorbar
    cbar_ax = fig.add_axes([0.7, 0.120, 0.25, 0.02])  # Adjust the position and size as needed
    
    # Create a mappable object for the colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="Greys_r")
    
    # Create the colorbar
    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pass Destination Volume', fontsize=14, fontproperties=font_prop, ha='left')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks([])  # Remove ticks
    cbar.outline.set_visible(False)  # Remove border
    cbar.solids.set(alpha=1)  # Adjust alpha value of the colorbar
    
    # Set the color of the label
    cbar.ax.xaxis.label.set_color('white')

    plt.figtext(0.05, 0.9, f"{selected_player} - Passes", fontproperties=font_prop_large, color='w', ha='left')
    plt.figtext(0.05, 0.85, first_game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(0.04, 0.165, "Regular", fontproperties=font_prop_small, color='#c791f2', ha='left')
    plt.figtext(0.04, 0.135, "Progressive", fontproperties=font_prop_small, color='#ff9d00', ha='left')
    plt.figtext(0.04, 0.105, "KeyPass", fontproperties=font_prop_small, color='#00aaff', ha='left')
    plt.figtext(0.04, 0.075, "Failed", fontproperties=font_prop_small, color='darkgrey', ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Whoscored.", fontproperties=font_prop_small, color='grey', ha='right')

    anim = FuncAnimation(fig, update, frames=len(pass_events_sorted), fargs=(ax, pitch, pass_events_sorted, '#ff9d00', '#c791f2', 'darkgrey', '#00aaff', "Greys_r"), interval=500, repeat=False, blit=True)

    # Save animation to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".gif") as temp_file:
        anim.save(temp_file.name, writer=PillowWriter(fps=2))
        temp_file.seek(0)
        gif = temp_file.read()

    # Display the animation in Streamlit
    gif_base64 = base64.b64encode(gif).decode('utf-8')
    st.markdown(f"![Animation](data:image/gif;base64,{gif_base64})")
