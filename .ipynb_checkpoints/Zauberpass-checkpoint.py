import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from mplsoccer import Pitch, FontManager
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from PIL import Image
from mplsoccer.utils import add_image
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define font properties and path effects
robotto_regular = FontManager()

path_eff = [path_effects.Stroke(linewidth=2.5, foreground='black'),
            path_effects.Normal()]


font_path = 'font1.ttf'
font_path2 = 'font2.ttf'
font_prop_title = fm.FontProperties(fname=font_path, size=40, weight='bold')
font_prop = fm.FontProperties(fname=font_path)
font_prop_title = fm.FontProperties(fname=font_path, size=40, weight='bold')
font_prop_medium = fm.FontProperties(fname=font_path, size=24, weight='bold')
font_prop_small = fm.FontProperties(fname=font_path, size=18, weight='bold')

def add_team_flag(fig, team_name, alpha=.75):
    """
    Adds a team flag to the figure if the flag image exists.
    """
    flag_image_path = f'flags/{team_name}.png'
    if os.path.isfile(flag_image_path):
        img = Image.open(flag_image_path).convert("RGBA")
        np_img = np.array(img)
        
        # Apply the alpha to the image
        np_img[:, :, 3] = (np_img[:, :, 3] * alpha).astype(np.uint8)
        
        img_ax = fig.add_axes([0.775, 0.8675, 0.07, 0.07])  # Adjusted coordinates for flag
        img_ax.imshow(np_img)
        img_ax.axis('off')  # Turn off axis
    else:
        st.warning(f"Flag image for {team_name} not found.")

def image_bg(img, fig):
    image_path = f"{img}.png"  # Assuming the background image is in PNG format
    image = Image.open(image_path)
    ax_image = add_image(image, fig, left=0, bottom=0, width=1, height=1)
    ax_image.set_zorder(0)

def add_colorbar(fig, cmap, position=[0.05, 0.15, 0.25, 0.02], labels=['Low', 'High'], font_prop=None):
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for standalone colorbar

    cbar_ax = fig.add_axes(position)  # Adjust the position and size
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xticks([0, 1])  # Only show low and high
    cbar.ax.set_xticklabels(labels, fontproperties=font_prop_small, color='grey')
    cbar.ax.tick_params(colors='grey')  # Set tick color to white

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

def draw_passmap(df, team, game_info, player, data_type_option):
    pass_events_sorted = df.sort_values(by=['minute', 'second'])
    
    pitch = Pitch(positional=True, positional_color='#3b3b3b', spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    

    image_bg("passmap_bg", fig)
    
    
    comp_clr = '#ff9d00'
    regular_clr = '#c791f2'
    failed_clr = 'darkgrey'
    key_pass_clr = '#00aaff'
    clr_map = "Greys_r"

    for _, row in pass_events_sorted.iterrows():
        draw_pass(ax, row, pitch, comp_clr, regular_clr, failed_clr, key_pass_clr)

    num_regular_passes = len(pass_events_sorted[pass_events_sorted['outcome_type'] == 'Successful'])
    num_failed_passes = len(pass_events_sorted[pass_events_sorted['outcome_type'] != 'Successful'])
    num_key_passes = len(pass_events_sorted[pass_events_sorted['qualifiers'].str.contains('KeyPass', na=False)])
    num_progressive_passes = sum(pass_events_sorted.apply(lambda row: is_long_pass(row['x'], row['end_x']), axis=1))


    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Passes", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(0.04, 0.165, f"Regular Passes: {num_regular_passes}", fontproperties=font_prop_small, color='#c791f2', ha='left')
    plt.figtext(0.04, 0.135, f"Progressive Passes: {num_progressive_passes}", fontproperties=font_prop_small, color='#ff9d00', ha='left')
    plt.figtext(0.04, 0.105, f"Key Passes: {num_key_passes}", fontproperties=font_prop_small, color='#00aaff', ha='left')
    plt.figtext(0.04, 0.075, f"Failed Passes: {num_failed_passes}", fontproperties=font_prop_small, color='darkgrey', ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')

    return fig


def draw_special_pass(ax, row, pitch, colors, lw_dict):
    x_start, y_start, x_end, y_end = row['x'], row['y'], row['end_x'], row['end_y']
    pass_color = colors[row['pass_type']]
    pass_width = lw_dict['successful'] if row['outcome_type'] == 'Successful' else lw_dict['unsuccessful']
    
    pitch.lines(xstart=x_start, ystart=y_start, xend=x_end, yend=y_end, 
                color=pass_color, lw=pass_width, zorder=4 if row['outcome_type'] == 'Successful' else 3, transparent=True, 
                alpha_start=0.75, alpha_end=0.01, ax=ax)

def draw_passmap_with_special_passes(df, team, game_info, player, data_type_option):
    pass_events_sorted = df.sort_values(by=['minute', 'second'])
    
    pitch = Pitch(positional=True, positional_color='#3b3b3b', spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()

    image_bg("passmap_bg", fig)
    
    special_pass_colors = {
        'BigChanceCreated': 'orangered',
        'FastBreak': 'deeppink',
        'Throughball': 'gold',
        'KeyPass': 'lightblue'
    }
    failed_clr = 'grey'
    lw_dict = {'successful': 4, 'unsuccessful': 3}

    special_passes_count = {key: 0 for key in special_pass_colors.keys()}
    special_passes_failed_count = {key: 0 for key in special_pass_colors.keys()}

    for _, row in pass_events_sorted.iterrows():
        if row['type'] == 'Pass':
            qualifiers = row['qualifiers']
            special_pass_types = ['BigChanceCreated', 'FastBreak', 'Throughball', 'KeyPass']
            if any(q in qualifiers for q in special_pass_types):
                row['pass_type'] = next(q for q in special_pass_types if q in qualifiers)
                if row['outcome_type'] == 'Successful':
                    special_passes_count[row['pass_type']] += 1
                    pass_color = special_pass_colors[row['pass_type']]
                else:
                    special_passes_failed_count[row['pass_type']] += 1
                    pass_color = failed_clr
                draw_special_pass(ax, row, pitch, {row['pass_type']: pass_color}, lw_dict)

    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Special Passes", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    
    for i, pass_type in enumerate(special_pass_colors.keys()):
        plt.figtext(0.04, 0.165 - i*0.03, f"{pass_type}:", fontproperties=font_prop_small, color=special_pass_colors[pass_type], ha='left')
        plt.figtext(0.2, 0.165 - i*0.03, f"{special_passes_count[pass_type]} successful", fontproperties=font_prop_small, color=special_pass_colors[pass_type], ha='left')
        plt.figtext(0.3, 0.165 - i*0.03, f"{special_passes_failed_count[pass_type]} failed", 
                    fontproperties=font_prop_small, color='grey', ha='left')
    
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')

    return fig


def draw_defensive_actions(df, team, game_info, player, data_type_option):
    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12))
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    defensive_actions = ['Tackle', 'Challenge', 'Interception', 'Foul']
    

    image_bg("passmap_bg", fig)

    if "Player" in data_type_option:
        team_data = df[(df['player'] == player) & (df['type'].isin(defensive_actions))]
    else:
        team_data = df[(df['team'] == team) & (df['type'].isin(defensive_actions))]

    bin_statistic = pitch.bin_statistic_positional(team_data.x, team_data.y, statistic='count',
                                               positional='full', normalize=True)
    pitch.heatmap_positional(bin_statistic, ax=ax, cmap='rocket', edgecolors='darkgrey')
    pitch.scatter(team_data.x, team_data.y, c='white', s=5, ax=ax)
    labels = pitch.label_heatmap(bin_statistic, color='lightgreen', fontsize=24,
                             ax=ax, ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff, rotation=0)
    
    # Draw a line at the average 'x' of the defensive actions
    mean_x = team_data['x'].mean()
    ax.axvline(x=mean_x, color='darkred', linestyle='-', linewidth=40, alpha=0.5, zorder=5)
    pitch.text(mean_x - 1, 50, 'Avg. Defensive Actions Height', color='w', ax=ax,fontproperties=font_prop_medium, va='bottom', ha='center', rotation=270, zorder=6)

    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Defenfive Actions Heatmap", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(0.95, 0.1, f"Defensive actions: tackles, interceptions, challanges, fouls. \nDirection of play from south to north. \nCoordinates from Opta.", ha='right', fontproperties=font_prop_small, color="grey")

    # Add the colorbar using the new function
    add_colorbar(fig, cmap='rocket', position=[0.05, 0.15, 0.25, 0.02], font_prop=font_prop_small)
    
    return fig

def draw_heatmap(df, team, game_info, player, data_type_option):
    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    

    image_bg("passmap_bg", fig)

    bin_statistic = pitch.bin_statistic_positional(df.x, df.y, statistic='count',
                                               positional='full', normalize=True)
    pitch.heatmap_positional(bin_statistic, ax=ax, cmap='rocket', edgecolors='darkgrey')
    pitch.scatter(df.x, df.y, c='white', s=5, ax=ax)
    labels = pitch.label_heatmap(bin_statistic, color='lightgreen', fontsize=24,
                             ax=ax, ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff, rotation=0)

    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Heatmap", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')

    
    add_colorbar(fig, cmap='rocket', position=[0.05, 0.15, 0.25, 0.02], font_prop=font_prop_small)
    
    return fig

def draw_takeons(df, team, game_info, player, data_type_option):
    pitch = Pitch(positional=True, positional_color='#3b3b3b', spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    

    image_bg("passmap_bg", fig)

    comp_clr = '#ff9d00'  # Define the color for successful take-ons
    count_s = 0
    count_f = 0
    for index, row in df.iterrows():
        if row['type'] == 'TakeOn' and row['outcome_type'] == 'Successful':
            pitch.scatter(row['x'], row['y'], color=comp_clr, marker='H', s=1200, zorder=3, ax=ax,edgecolor='black', linewidth=0, alpha=.9)
            count_s+=1
        elif row['type'] == 'TakeOn' and row['outcome_type'] == 'Unsuccessful':
            pitch.scatter(row['x'], row['y'], color='grey', marker='H', s=1200, zorder=3, ax=ax,edgecolor='grey', linewidth=0, alpha=.3)
            count_f+=1
    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Take-ons", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(0.04, 0.165, f"Completed: {count_s}", fontproperties=font_prop_small, color=comp_clr, ha='left')
    plt.figtext(0.04, 0.135, f"Failed: {count_f}", fontproperties=font_prop_small, color='darkgrey', ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')
    return fig

def draw_pass_receptions(df, team, game_info, player, data_type_option):
    required_columns = ['minute', 'second', 'game_id', 'x', 'y', 'end_x', 'end_y', 'player', 'type', 'outcome_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in the DataFrame: {missing_columns}")
    
    pitch = Pitch(spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()

    image_bg("passmap_bg", fig)
    
    # Filter for the specific team and successful passes
    df = df[(df['team'] == team) & (df['type'] == 'Pass') & (df['outcome_type'] == 'Successful')]

    if 'Player' in data_type_option:
        # Identify pass receptions for the specific player
        df['passer'] = df['player']
        df['recipient'] = df['passer'].shift(-1)
        df['time'] = df['minute'] * 60 + df['second']
        df_recipient = df[(df['type'] == 'Pass') & (df['outcome_type'] == 'Successful') & (df['recipient'] == player)]

        # Ensure the game_id matches for consecutive rows
        condition = df['game_id'] == df['game_id'].shift(-1)
        end_x = df_recipient['end_x']
        end_y = df_recipient['end_y']
    else:
        end_x = df['end_x']
        end_y = df['end_y']
    
    bin_statistic = pitch.bin_statistic_positional(end_x, end_y, statistic='count',
                                                   positional='full', normalize=True)
    pitch.heatmap_positional(bin_statistic, ax=ax, cmap='rocket', edgecolors='darkgrey')
    pitch.scatter(end_x, end_y, c='white', s=5, ax=ax)
    labels = pitch.label_heatmap(bin_statistic, color='lightgreen', fontsize=24,
                                 ax=ax, ha='center', va='center',
                                 str_format='{:.0%}', path_effects=path_eff, rotation=0)

    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Pass Receptions", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')
    
    add_colorbar(fig, cmap='rocket', position=[0.05, 0.15, 0.25, 0.02], font_prop=font_prop_small)
    
    return fig

def find_top_pass_clusters(df, num_clusters=10, top_n=3):
    # Filter successful passes
    passes = df[(df['type'] == 'Pass') & (df['outcome_type'] == 'Successful')]

    if passes.empty:
        return None, None

    # Remove rows with non-finite values in the necessary columns
    passes = passes[['x', 'y', 'end_x', 'end_y']].dropna()

    # Ensure columns are numeric
    required_columns = ['x', 'y', 'end_x', 'end_y']
    if not all(pd.api.types.is_numeric_dtype(passes[col]) for col in required_columns):
        raise ValueError("One or more columns are not numeric.")

    # Adjust the number of clusters if necessary
    actual_num_clusters = min(num_clusters, len(passes))
    if actual_num_clusters < 1:
        raise ValueError("Not enough data points to form any clusters.")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=100).fit(passes)
    passes['cluster'] = kmeans.labels_

    # Find top clusters by number of passes
    top_clusters = passes['cluster'].value_counts().nlargest(top_n).index

    cluster_info = []
    for cluster in top_clusters:
        cluster_data = passes[passes['cluster'] == cluster]
        avg_start_x = cluster_data['x'].mean()
        avg_start_y = cluster_data['y'].mean()
        avg_end_x = cluster_data['end_x'].mean()
        avg_end_y = cluster_data['end_y'].mean()
        cluster_info.append({
            'cluster': cluster,
            'avg_start_x': avg_start_x,
            'avg_start_y': avg_start_y,
            'avg_end_x': avg_end_x,
            'avg_end_y': avg_end_y,
            'count': len(cluster_data)
        })

    return passes, cluster_info

def draw_pass_clusters(passes, cluster_info, team, game_info, player, data_type_option):
    pitch = Pitch(positional=True, positional_color='#3b3b3b',spot_type='square', spot_scale=0.01, pitch_type='wyscout', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    plt.gca().invert_yaxis()
    
    image_bg("passmap_bg", fig)
    
    colors = ['lightgreen', 'deeppink', 'royalblue']
    arrow_color = 'gold'
    arrow_length_scale = 2

    for i, cluster in enumerate(cluster_info):
        cluster_data = passes[passes['cluster'] == cluster['cluster']]
        pitch.lines(xstart=cluster_data['x'], ystart=cluster_data['y'], xend=cluster_data['end_x'], 
                    yend=cluster_data['end_y'], color=colors[i], lw=3, zorder=3, transparent=True, alpha_start=0.25, alpha_end=0.01, ax=ax)
        
        # Plot the average direction arrow
        avg_x = cluster['avg_start_x']
        avg_y = cluster['avg_start_y']
        delta_x = (cluster['avg_end_x'] - cluster['avg_start_x']) * arrow_length_scale
        delta_y = (cluster['avg_end_y'] - cluster['avg_start_y']) * arrow_length_scale
        pitch.arrows(avg_x, avg_y, avg_x + delta_x, avg_y + delta_y, color=colors[i], ax=ax, 
                     width=7, headwidth=5, headlength=5, path_effects=path_eff, zorder=4)
    
    # Load your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Top 3 Pass Clusters", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(0.04, 0.165, f"Cluster rank 1", fontproperties=font_prop_small, color=colors[0], ha='left')
    plt.figtext(0.04, 0.135, f"Cluster rank 2", fontproperties=font_prop_small, color=colors[1], ha='left')
    plt.figtext(0.04, 0.105, f"Cluster rank 3", fontproperties=font_prop_small, color=colors[2], ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')
    return fig

def load_data(tournament):
    data_sources = {
        "Euro 2024": "https://drive.google.com/uc?export=download&id=1-IJfIqkYv39CRoSLgXdMd3QZrP0a9ViL",
        "Euro 2024 Spadl": "https://drive.google.com/uc?export=download&id=1-jL8HwJQRC13qgHzks4Y5Kzi8wbwH6Wv",
        "Copa America 2024": "https://drive.google.com/uc?export=download&id=1-ehpOjBEsZKRT5el17KaH7f5_QouaWum",
        "Copa America 2024 Spadl": "https://drive.google.com/uc?export=download&id=1-mrCWGOphVYR65mQzRetL5AvcKvoPjvn"
    }
    url = data_sources[tournament]
    df = pd.read_csv(url)
    return df

def load_spadl_data(tournament):
    spadl_data_sources = {
        "Euro 2024": "https://drive.google.com/uc?export=download&id=1-jL8HwJQRC13qgHzks4Y5Kzi8wbwH6Wv",
        "Copa America 2024": "https://drive.google.com/uc?export=download&id=1-mrCWGOphVYR65mQzRetL5AvcKvoPjvn"
    }
    url = spadl_data_sources[tournament]
    df = pd.read_csv(url)
    return df

def draw_carries(df, team, game_info, player, data_type_option):
    required_columns = ['type_id', 'start_x', 'start_y', 'end_x', 'end_y', 'game_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in the DataFrame: {missing_columns}")

    pitch = Pitch(positional=True, positional_color='#3b3b3b', spot_type='square', spot_scale=0.01, pitch_type='uefa', line_color='lightgrey', linewidth=4, line_zorder=2, pitch_color='None')
    fig, ax = pitch.draw(figsize=(12, 12), constrained_layout=True)
    fig.set_facecolor('black')
    ax.patch.set_facecolor('None')
    ax.set_zorder(1)
    
    # Add background image
    image_bg("passmap_bg", fig)
    
    prg = 0
    reg = 0
    
    if "Player" in data_type_option:
        df = df[df['player'] == player]
    else:
        df = df[df['team'] == team]
    
    for x in range(len(df)):
        if df['type_id'].iloc[x] == 21:
            xs = df['start_x'].iloc[x]
            ys = df['start_y'].iloc[x]
            xe = df['end_x'].iloc[x]
            ye = df['end_y'].iloc[x]
            dist = np.sqrt((xe - xs) ** 2)

            if (xe > 50 and (xe - xs) >= 10) or (82 <= xe <= 100 and 17 <= ye <= 83 and xe > xs):
                pitch.lines(xstart=xs, ystart=ys, xend=xe, yend=ye, color="#fca103", lw=5, zorder=3, transparent=True, alpha_start=.01, alpha_end=1, ax=ax)
                prg += 1            
            else:
                pitch.lines(xstart=xs, ystart=ys, xend=xe, yend=ye, color="grey", lw=4, zorder=2, transparent=True, alpha_start=.01, alpha_end=0.75, ax=ax)
                reg += 1
    
    # Load and add your image
    image_path = 'blogo.png'  # Replace with the path to your image
    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.85, 0.85, 0.1, 0.1])  # Example: [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.axis('off')  # Turn off axis
    
    add_team_flag(fig, team, alpha=0.75)
    
    plt.figtext(0.05, 0.9, f"{player if 'Player' in data_type_option else team} - Carries", fontproperties=font_prop_title, color='w', ha='left')
    plt.figtext(0.05, 0.85, game_info, fontproperties=font_prop_medium, color='#2af5bf', ha='left')
    plt.figtext(0.04, 0.165, f"Progressive Carries: {prg}", fontproperties=font_prop_small, color="#fca103", ha='left')
    plt.figtext(0.04, 0.135, f"Regular Carries: {reg}", fontproperties=font_prop_small, color="grey", ha='left')
    plt.figtext(.95, 0.175, "Direction of play from left to right. Coordinates from Opta.", fontproperties=font_prop_small, color='grey', ha='right')
    
    return fig

# Load and resize the logo
image = Image.open("zplogo2.png")
image = image.resize((100, 100))  # Resize to 100x100 pixels

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
    game_info = f"All Games in {selected_tournament}"
else:
    match_df = filtered_df_game
    game_info_full = match_df.iloc[0]['game']
    game_info = game_info_full.split(' ', 1)[1]+f" in {selected_tournament}" if len(match_df) > 0 else 'Game Information Not Available'

if "Player" in data_type_option:
    filtered_df = match_df[match_df['player'] == selected_player]
else:
    filtered_df = match_df

compare = st.checkbox("Compare")

if compare:
    selected_tournament_compare = st.selectbox("Select Tournament for Comparison", tournaments, key='compare_tournament')
    df_compare = load_data(selected_tournament_compare)

    data_type_option_compare = st.radio("Select Data Type for Comparison", ["Player - Match by Match", "Player - All Games", "Team - Match by Match", "Team - All Games"], key='compare_data_type')

    teams_compare = sorted(df_compare['team'].unique())
    selected_team_compare = st.selectbox("Select Team for Comparison", teams_compare, index=0, key='compare_team')

    filtered_df_team_compare = df_compare[df_compare['team'] == selected_team_compare]
    matches_compare = sorted(filtered_df_team_compare['game'].unique())
    selected_match_compare = st.selectbox("Select Match for Comparison", matches_compare, index=0, disabled=("All Games" in data_type_option_compare), key='compare_match')

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

if st.button("Full Analysis"):
    col1, col2 = st.columns(2)

    # Load SPADL data for Carries
    filtered_df_spadl = load_spadl_data(selected_tournament)
    if compare:
        filtered_df_compare_spadl = load_spadl_data(selected_tournament_compare)

    # Load appropriate data for other analyses
    if "Player" in data_type_option:
        filtered_df = match_df[match_df['player'] == selected_player]
    else:
        filtered_df = match_df
    if compare:
        if "Player" in data_type_option_compare:
            filtered_df_compare = match_df_compare[match_df_compare['player'] == selected_player_compare]
        else:
            filtered_df_compare = match_df_compare

    with col1:
        fig1_heatmap = draw_heatmap(filtered_df, selected_team, game_info, selected_player, data_type_option)
        fig1_passmap = draw_passmap(filtered_df, selected_team, game_info, selected_player, data_type_option)
        fig1_special_passes = draw_passmap_with_special_passes(filtered_df, selected_team, game_info, selected_player, data_type_option)
        fig1_pass_receptions = draw_pass_receptions(filtered_df, selected_team, game_info, selected_player, data_type_option)
        fig1_defensive_actions = draw_defensive_actions(filtered_df, selected_team, game_info, selected_player, data_type_option)
        fig1_takeons = draw_takeons(filtered_df, selected_team, game_info, selected_player, data_type_option)
        fig1_carries = draw_carries(filtered_df_spadl, selected_team, game_info, selected_player, data_type_option)
        
        passes, cluster_info = find_top_pass_clusters(filtered_df)
        if passes is not None and cluster_info is not None:
            fig1_clusters = draw_pass_clusters(passes, cluster_info, selected_team, game_info, selected_player, data_type_option)
            st.pyplot(fig1_clusters)
        else:
            st.write("No successful passes found for clustering.")
        
        st.pyplot(fig1_heatmap)
        st.pyplot(fig1_passmap)
        st.pyplot(fig1_special_passes)
        st.pyplot(fig1_pass_receptions)
        st.pyplot(fig1_defensive_actions)
        st.pyplot(fig1_takeons)
        st.pyplot(fig1_carries)

    if compare:
        with col2:
            fig2_heatmap = draw_heatmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            fig2_passmap = draw_passmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            fig2_special_passes = draw_passmap_with_special_passes(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            fig2_pass_receptions = draw_pass_receptions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            fig2_defensive_actions = draw_defensive_actions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            fig2_takeons = draw_takeons(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            fig2_carries = draw_carries(filtered_df_compare_spadl, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)

            passes_compare, cluster_info_compare = find_top_pass_clusters(filtered_df_compare)
            if passes_compare is not None and cluster_info_compare is not None:
                fig2_clusters = draw_pass_clusters(passes_compare, cluster_info_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
                st.pyplot(fig2_clusters)
            else:
                st.write("No successful passes found for clustering.")
                
            st.pyplot(fig2_heatmap)
            st.pyplot(fig2_passmap)
            st.pyplot(fig2_special_passes)
            st.pyplot(fig2_pass_receptions)
            st.pyplot(fig2_defensive_actions)
            st.pyplot(fig2_takeons)
            st.pyplot(fig2_carries)

if st.button("Top 3 Pass Clusters"):
    col1, col2 = st.columns(2)

    with col1:
        passes, cluster_info = find_top_pass_clusters(filtered_df)
        if passes is not None and cluster_info is not None:
            fig1 = draw_pass_clusters(passes, cluster_info, selected_team, game_info, selected_player, data_type_option)
            st.pyplot(fig1)
        else:
            st.write("No successful passes found for clustering.")

    if compare:
        with col2:
            passes_compare, cluster_info_compare = find_top_pass_clusters(filtered_df_compare)
            if passes_compare is not None and cluster_info_compare is not None:
                fig2 = draw_pass_clusters(passes_compare, cluster_info_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
                st.pyplot(fig2)
            else:
                st.write("No successful passes found for clustering.")

if st.button("Passmap"):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = draw_passmap(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            fig2 = draw_passmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)

if st.button("Special Passes"):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = draw_passmap_with_special_passes(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            fig2 = draw_passmap_with_special_passes(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)

if st.button("TakeOns"):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = draw_takeons(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            fig2 = draw_takeons(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)

if st.button("Carries"):
    col1, col2 = st.columns(2)

    with col1:
        filtered_df = load_spadl_data(selected_tournament)
        fig1 = draw_carries(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            filtered_df_compare = load_spadl_data(selected_tournament_compare)
            fig2 = draw_carries(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)

            
if st.button("Heatmap"):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = draw_heatmap(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            fig2 = draw_heatmap(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)

if st.button("Pass Reception"):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = draw_pass_receptions(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            fig2 = draw_pass_receptions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)

if st.button("Defensive Actions"):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = draw_defensive_actions(filtered_df, selected_team, game_info, selected_player, data_type_option)
        st.pyplot(fig1)

    if compare:
        with col2:
            fig2 = draw_defensive_actions(filtered_df_compare, selected_team_compare, game_info_compare, selected_player_compare, data_type_option_compare)
            st.pyplot(fig2)
