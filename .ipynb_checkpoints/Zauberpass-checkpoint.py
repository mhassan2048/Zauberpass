import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from PIL import Image
from mplsoccer.utils import add_image

# Add custom font
font_path = 'DIN-Condensed-Bold.ttf'
font_prop = fm.FontProperties(fname=font_path)
font_prop_large = fm.FontProperties(fname=font_path, size=48, weight='bold')
font_prop_medium = fm.FontProperties(fname=font_path, size=24, weight='bold')
font_prop_small = fm.FontProperties(fname=font_path, size=20, weight='bold')

def image_bg(img_path, fig):
    image = Image.open(img_path)
    ax_image = add_image(image, fig, left=0, bottom=0, width=1, height=1)
    ax_image.set_zorder(0)

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

def draw_passmap(df, team, game_info, player, data_type_option, image_path):
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    pitch = Pitch(positional=True, positional_color='#3b3b3b', spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    pitch.draw(ax=ax)
    ax.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    
    image_bg(image_path, fig)
    
    comp_clr = '#ff9d00'
    regular_clr = '#c791f2'
    failed_clr = 'darkgrey'
    key_pass_clr = '#00aaff'

    for _, row in df.iterrows():
        draw_pass(ax, row, pitch, comp_clr, regular_clr, failed_clr, key_pass_clr)

    num_regular_passes = len(df[df['outcome_type'] == 'Successful'])
    num_failed_passes = len(df[df['outcome_type'] != 'Successful'])
    num_key_passes = len(df[df['qualifiers'].str.contains('KeyPass', na=False)])
    num_progressive_passes = sum(df.apply(lambda row: is_long_pass(row['x'], row['end_x']), axis=1))

    ax.set_title(f"{player if 'Player' in data_type_option else team} - Passes", fontproperties=font_prop_large, color='w', loc='left')
    ax.text(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left', transform=ax.transAxes)
    ax.text(0.04, 0.165, f"Regular Passes: {num_regular_passes}", fontproperties=font_prop_small, color='#c791f2', ha='left', transform=ax.transAxes)
    ax.text(0.04, 0.135, f"Progressive Passes: {num_progressive_passes}", fontproperties=font_prop_small, color='#ff9d00', ha='left', transform=ax.transAxes)
    ax.text(0.04, 0.105, f"Key Passes: {num_key_passes}", fontproperties=font_prop_small, color='#00aaff', ha='left', transform=ax.transAxes)
    ax.text(0.04, 0.075, f"Failed Passes: {num_failed_passes}", fontproperties=font_prop_small, color='darkgrey', ha='left', transform=ax.transAxes)
    ax.text(.95, 0.175, "Direction of play from left to right. Coordinates from Whoscored.", fontproperties=font_prop_small, color='grey', ha='right', transform=ax.transAxes)

    return fig


def draw_heatmap(df, team, game_info, player, data_type_option, image_path):
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    pitch.draw(ax=ax)
    ax.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    
    image_bg(image_path, fig)
    
    bs = pitch.bin_statistic(df.x, df.y, bins=(48, 32))
    pitch.heatmap(bs, ax=ax, edgecolor='black', linewidth=4, cmap='magma')

    ax.set_title(f"{player if 'Player' in data_type_option else team} - Heatmap", fontproperties=font_prop_large, color='w', loc='left')
    ax.text(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left', transform=ax.transAxes)
    ax.text(.95, 0.175, "Direction of play from left to right. Coordinates from Whoscored.", fontproperties=font_prop_small, color='grey', ha='right', transform=ax.transAxes)

    return fig

def draw_defensive_actions(df, team, game_info, player, data_type_option, image_path):
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    pitch.draw(ax=ax)
    ax.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    defensive_actions = ['Tackle', 'Challenge', 'Interception', 'Foul']
    
    image_bg(image_path, fig)
    
    if "Player" in data_type_option:
        team_data = df[(df['player'] == player) & (df['type'].isin(defensive_actions))]
    else:
        team_data = df[(df['team'] == team) & (df['type'].isin(defensive_actions))]

    bs = pitch.bin_statistic(team_data.x, team_data.y, bins=(48, 32))
    pitch.heatmap(bs, ax=ax, edgecolor='black', linewidth=4, cmap='rocket')
    
    mean_y = team_data['y'].mean()
    ax.axvline(x=mean_y, color='lightgreen', linestyle='-', linewidth=40, alpha=0.3)
    pitch.text(mean_y - 1, 50, 'Avg. Defensive Actions Height', color='w', ax=ax,fontproperties=font_prop_medium, va='bottom', ha='center', rotation=270)

    ax.set_title(f"{player if 'Player' in data_type_option else team} - Defensive Actions Heatmap", fontproperties=font_prop_large, color='w', loc='left')
    ax.text(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left', transform=ax.transAxes)
    ax.text(0.5, 0.08, "Defensive actions: tackles, interceptions, challenges, fouls. \nDirection of play from south to north. \nCoordinates from Whoscored.", ha='center', fontproperties=font_prop_small, color="grey", transform=ax.transAxes)

    return fig

def draw_takeons(df, team, game_info, player, data_type_option, image_path):
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    pitch = Pitch(positional=True, positional_color='#3b3b3b', spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    pitch.draw(ax=ax)
    ax.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    
    image_bg(image_path, fig)
    
    comp_clr = '#ff9d00'
    count_s = 0
    count_f = 0
    for index, row in df.iterrows():
        if row['type'] == 'TakeOn' and row['outcome_type'] == 'Successful':
            pitch.scatter(row['x'], row['y'], color=comp_clr, marker='H', s=1200, zorder=3, ax=ax, edgecolor='black', linewidth=0, alpha=.9)
            count_s += 1
        elif row['type'] == 'TakeOn' and row['outcome_type'] == 'Unsuccessful':
            pitch.scatter(row['x'], row['y'], color='grey', marker='H', s=1200, zorder=3, ax=ax, edgecolor='grey', linewidth=0, alpha=.3)

    ax.set_title(f"{player if 'Player' in data_type_option else team} - Take-ons", fontproperties=font_prop_large, color='w', loc='left')
    ax.text(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left', transform=ax.transAxes)
    ax.text(0.04, 0.165, f"Completed: {count_s}", fontproperties=font_prop_small, color=comp_clr, ha='left', transform=ax.transAxes)
    ax.text(0.04, 0.135, f"Failed: {count_f}", fontproperties=font_prop_small, color='darkgrey', ha='left', transform=ax.transAxes)
    ax.text(.95, 0.175, "Direction of play from left to right. Coordinates from Whoscored.", fontproperties=font_prop_small, color='grey', ha='right', transform=ax.transAxes)

    return fig

def draw_pass_receptions(df, team, game_info, player, data_type_option, image_path):
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    pitch.draw(ax=ax)
    ax.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    
    image_bg(image_path, fig)
    
    bs = pitch.bin_statistic(df.end_x, df.end_y, bins=(48, 32))
    pitch.heatmap(bs, ax=ax, edgecolor='black', linewidth=4, cmap='magma')

    ax.set_title(f"{player if 'Player' in data_type_option else team} - Pass Receptions", fontproperties=font_prop_large, color='w', loc='left')
    ax.text(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left', transform=ax.transAxes)
    ax.text(.95, 0.175, "Direction of play from left to right. Coordinates from Whoscored.", fontproperties=font_prop_small, color='grey', ha='right', transform=ax.transAxes)

    return fig

def load_data(tournament):
    data_sources = {
        "Euro 2024": "https://drive.google.com/uc?export=download&id=1-IJfIqkYv39CRoSLgXdMd3QZrP0a9ViL",
        "Copa America 2024": "https://drive.google.com/uc?export=download&id=1-ehpOjBEsZKRT5el17KaH7f5_QouaWum"
    }
    url = data_sources[tournament]
    df = pd.read_csv(url)
    return df

# Load and resize the logo
image = Image.open("zplogo.png")
image = image.resize((100, 100))

# Display the logo and title
st.image(image, use_column_width=False)
st.title("Zauberpass by The Real Deal")

# Add Tournament Dropdown
tournaments = ["Euro 2024", "Copa America 2024"]
selected_tournament = st.selectbox("Select Tournament", tournaments)

# Load data based on selected tournament
df = load_data(selected_tournament)

# Add Radio Buttons for Data Type Selection
data_type_option = st.radio("Select Data Type", ["Player - Match by Match", "Player - All Games", "Team - Match by Match", "Team - All Games"])

# Extract unique teams
teams = sorted(df['team'].unique())
selected_team = st.selectbox("Select Team", teams, index=0)

# Filter matches based on the selected team
filtered_df_team = df[df['team'] == selected_team]
matches = sorted(filtered_df_team['game'].unique())
selected_match = st.selectbox("Select Match", matches, index=0, disabled=("All Games" in data_type_option))

# Filter players based on the selected team and game
filtered_df_game = filtered_df_team[filtered_df_team['game'] == selected_match]
players = sorted(filtered_df_game['player'].dropna().unique())
selected_player = st.selectbox("Select Player", players, index=0, disabled=("Team" in data_type_option))

if "All Games" in data_type_option:
    match_df = filtered_df_team
    game_info = f"All Games - {selected_tournament}"
else:
    match_df = filtered_df_game
    game_info = match_df.iloc[0]['game'] if len(match_df) > 0 else 'Game Information Not Available'

if "Player" in data_type_option:
    filtered_df = match_df[match_df['player'] == selected_player]
else:
    filtered_df = match_df

# Add options for the second set of data for comparison
compare = st.checkbox("Compare with another set of data")

if compare:
    # Add Tournament Dropdown for comparison
    selected_tournament_compare = st.selectbox("Select Tournament for Comparison", tournaments, key='compare_tournament')
    df_compare = load_data(selected_tournament_compare)

    # Add Radio Buttons for Data Type Selection for comparison
    data_type_option_compare = st.radio("Select Data Type for Comparison", ["Player - Match by Match", "Player - All Games", "Team - Match by Match", "Team - All Games"], key='compare_data_type')

    # Extract unique teams for comparison
    teams_compare = sorted(df_compare['team'].unique())
    selected_team_compare = st.selectbox("Select Team for Comparison", teams_compare, index=0, key='compare_team')

    # Filter matches based on the selected team for comparison
    filtered_df_team_compare = df_compare[df_compare['team'] == selected_team_compare]
    matches_compare = sorted(filtered_df_team_compare['game'].unique())
    selected_match_compare = st.selectbox("Select Match for Comparison", matches_compare, index=0, disabled=("All Games" in data_type_option_compare), key='compare_match')

    # Filter players based on the selected team and game for comparison
    filtered_df_game_compare = filtered_df_team_compare[filtered_df_team_compare['game'] == selected_match_compare]
    players_compare = sorted(filtered_df_game_compare['player'].dropna().unique())
    selected_player_compare = st.selectbox("Select Player for Comparison", players_compare, index=0, disabled=("Team" in data_type_option_compare), key='compare_player')

    if "All Games" in data_type_option_compare:
        match_df_compare = filtered_df_team_compare
        game_info_compare = f"All Games - {selected_tournament_compare}"
    else:
        match_df_compare = filtered_df_game_compare
        game_info_compare = match_df_compare.iloc[0]['game'] if len(match_df_compare) > 0 else 'Game Information Not Available'

    if "Player" in data_type_option_compare:
        filtered_df_compare = match_df_compare[match_df_compare['player'] == selected_player_compare]
    else:
        filtered_df_compare = match_df_compare

# Add buttons for new features
if st.button("Full Analysis"):
    fig1_heatmap = draw_heatmap(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    fig1_passmap = draw_passmap(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    fig1_pass_receptions = draw_pass_receptions(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    fig1_defensive_actions = draw_defensive_actions(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    fig1_takeons = draw_takeons(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    
    st.pyplot(fig1_heatmap)
    st.pyplot(fig1_passmap)
    st.pyplot(fig1_pass_receptions)
    st.pyplot(fig1_defensive_actions)
    st.pyplot(fig1_takeons)

    if compare:
        fig2_heatmap = draw_heatmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        fig2_passmap = draw_passmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        fig2_pass_receptions = draw_pass_receptions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        fig2_defensive_actions = draw_defensive_actions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        fig2_takeons = draw_takeons(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        
        st.pyplot(fig2_heatmap)
        st.pyplot(fig2_passmap)
        st.pyplot(fig2_pass_receptions)
        st.pyplot(fig2_defensive_actions)
        st.pyplot(fig2_takeons)

if st.button("TakeOns"):
    fig1 = draw_takeons(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    st.pyplot(fig1)

    if compare:
        fig2 = draw_takeons(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        st.pyplot(fig2)

if st.button("Heatmap"):
    fig1 = draw_heatmap(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    st.pyplot(fig1)

    if compare:
        fig2 = draw_heatmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        st.pyplot(fig2)

if st.button("Pass Reception"):
    fig1 = draw_pass_receptions(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    st.pyplot(fig1)

    if compare:
        fig2 = draw_pass_receptions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        st.pyplot(fig2)

if st.button("Defensive Actions"):
    fig1 = draw_defensive_actions(filtered_df, selected_team, game_info, selected_player, data_type_option, "path/to/background_image1.png")
    st.pyplot(fig1)

    if compare:
        fig2 = draw_defensive_actions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare, "path/to/background_image2.png")
        st.pyplot(fig2)
