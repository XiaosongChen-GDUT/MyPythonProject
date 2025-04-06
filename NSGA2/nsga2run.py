import pandas as pd
from matplotlib import cm

import matplotlib.animation as animation
import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import chain
import random
import matplotlib.pyplot as plt
import math
import folium
from matplotlib import colormaps as cm
from scipy import stats
import time
import os
from selenium import webdriver
from tqdm import tqdm
from selenium.webdriver.firefox.options import Options
import imageio
from PIL import Image, ImageDraw, ImageFont
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import imageio.v2 as imageio

# options = Options()
# service = ChromeService(executable_path=r"C:\Users\12806\Desktop\Google Chrome.lnk")
# driver = webdriver.Chrome(service=service, options=options)

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

from NSGA2.Evolution import Evolution
from NSGA2.Problem import Problem

# file_path1 = '../data/distance.xlsx'
# file_path2 = '../data/capacity.xlsx'
# file_path3 = '../data/population.xlsx'
file_path1 = 'data/distance.xlsx'
file_path2 = 'data/capacity.xlsx'
file_path3 = 'data/population.xlsx'

df_distance = pd.read_excel(file_path1)
df_capacity = pd.read_excel(file_path2)
df_population = pd.read_excel(file_path3)
df_walking = df_distance.copy()  # Start with a copy of the original
df_driving = df_distance.copy()

for column in df_distance.columns:
    if column.startswith('sh'):
        # Split the column
        split_result = df_distance[column].str.split('/', expand=True)
        # Assign values to new DataFrames
        df_walking[column] = split_result[0]  # Walking distances
        if 1 in split_result.columns:
            df_driving[column] = split_result[1]  # Driving distances
        else:
            df_driving[column] = None  # Handle missing driving distances

building_blocks = {}
shelters = {}
for index, row in df_capacity.iterrows():
    shelter_name = row['Shelter']
    capacity = int(row['Capacity'])
    lat_lng = row['Lat_Lng']
    lat, lng = map(float, row['Lat_Lng'].split(', '))
    shelters[index] = {'capacity': capacity, 'lat': lat, 'lng': lng}
for index, row in df_population.iterrows():
    block_name = row['Building_Block']
    population = int(row['Population'])
    lat, lng = map(float, row['Lat_Lng'].split(', '))
    building_blocks[index] = {'population': population, 'lat': lat, 'lng': lng}
print(len(shelters))
print(len(building_blocks))

def get_color(value, min_value, max_value):
    """
    Maps a value to a color in a green-to-red colormap.
    """
    if value > max_value:
        return '#{:02x}{:02x}{:02x}'.format(0, 0, 0)
    norm_value = (value - min_value) / (max_value - min_value)
    cmap = cm.get_cmap('RdYlGn_r')  # Reverse green-yellow-red colormap
    color = cmap(norm_value)[:3]  # Convert to RGB tuple
    return '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

def map_animation(individual, filename):
    number_of_shelters = len(shelters)
    number_of_blocks = len(building_blocks)
    shelter_overload = [0.0 for _ in range(number_of_shelters)]
    m = folium.Map(location=[-25.8900, 32.6158], zoom_start=11.5)
    for i in range (len(individual)):
        building_block = building_blocks[i]
        shelter = shelters[individual[i] - 1]
        shelter_overload[individual[i] - 1] += float(df_population.iloc[i,1])
        point_1 = [building_block['lat'], building_block['lng']]
        point_2 = [shelter['lat'], shelter['lng']]
        folium.PolyLine([point_1, point_2], color="black", weight=1).add_to(m)

    color = []
    icon_props = dict(prefix='fa', icon='user', color='black')
    for i in range (number_of_shelters):
        cap = df_capacity.iloc[i,1]
        shelter_overload[i] = shelter_overload[i] - float(cap)
        overload = shelter_overload[i]
        shelter = shelters[i]
        max_value = float(cap) * 3/2
        min_value = -1 * float(cap)
        color = get_color(overload, min_value, max_value)
        text = str(overload)
        folium.CircleMarker(
            location = [shelter['lat'], shelter['lng']],
            radius =  shelter['capacity'] / 50,  # radius in pixels
            color = color,
            fill = True,
            fill_color = color,
            fill_opacity = 1,
            tooltip = f"{overload}",
            icon=folium.Icon(**icon_props, text=text, icon_size=(2, 2))
        ).add_to(m)
    for i in range(number_of_blocks):
        building_block = building_blocks[i]
        folium.CircleMarker(
            location = [building_block['lat'], building_block['lng']],
            radius = building_block['population'] / 50,  # radius in pixels
            color = 'blue',
            fill = True,
            fill_color ='blue',
            fill_opacity = 1
        ).add_to(m)
    # Save the map as an image
    m.save(filename)
    return filename

def convert_html_to_png(path, pare_to_number,):
    options = Options()
    # for i in range(numb_iteration): 
    filename = path
    mapURL = 'file://{0}/{1}'.format(os.getcwd(),filename)
    driver = webdriver.Firefox(options=options)
    driver.get(mapURL)
    time.sleep(1)
    filename2 = f'map_animation_{pare_to_number}.png'
    driver.save_screenshot(filename2)
    driver.quit()
    write_on_picture(pare_to_number,filename2)

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def write_on_picture(pare_to_number,filename2):
    font_path = "Candara.ttf"
    font_size = 60
    text_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size)
    image_path = filename2
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    text = f" number in front : {pare_to_number}"
    text_width, text_height = textsize(text, font=font)
    x = 500  # Adjust horizontal position
    y = 80  # Adjust vertical position
    draw.text((x, y), text, fill=text_color, font=font)
    # Save the modified image
    img.save(filename2)

def make_animation(Front):  #
    #初始化文件名列表
    filenames = []
    for i in range(len(Front)):
        filenames.append(f"map_animation_{i}.png")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        print(f"Image {filename} shape: {imageio.imread(filename).shape}")
    # Create a GIF animation (adjust duration as needed)
    imageio.mimsave('map_animation.gif', images, fps = 3)

def make_animation2():
    filenames = []
    for i in range (500):
        filenames.append(f"pareToFront_of_generation_{i}.png")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        print(f"Image {filename} shape: {imageio.imread(filename).shape}")
    # Create a GIF animation (adjust duration as needed)
    imageio.mimsave('pare_to_animation.gif', images, fps = 9)

#目标函数
# 从数据框中获取避难所和建筑区块的数量
number_of_shelters = len(df_capacity)
number_of_blocks = len(df_population)

def fdistance(x):
    y = 0.0
    for i in range (len(x)):
        y += float(df_population.iloc[i,1]) * float(df_walking.iloc[i,x[i]])
    return y

def fcapacity(x):
    y = 0.0
    shelter_population = [0.0 for _ in range (number_of_shelters)]
    for i in range (len(x)):
        shelter_population[x[i] - 1] += int(df_population.iloc[i,1])
    for i in range (number_of_shelters):
        y += abs((shelter_population[i] / int(df_capacity.iloc[i,1])) - 1)
    return y

problem = Problem(num_of_variables = number_of_blocks, objectives = [fdistance, fcapacity], variables_range = number_of_shelters)
evo = Evolution(problem, num_of_generations = 500, mutation_param =  0.13630410571719337, tournament_prob = 1.0, num_of_individuals = 170,  num_of_tour_particips = 10, use_threshold_flag = False)
Front ,_ = evo.evolve()
Fcapacity = []
Fdistance = []
pare_to = [[]]
counter = 0
plt.xlabel('Fcapacity', fontsize= 15)
plt.ylabel('Fdistance', fontsize= 15)
# for i in Front:
#     Fcapacity.append(i.objectives[1])
#     Fdistance.append(i.objectives[0])
#     pare_to.append([i.objectives[0],i.objectives[1]])
#     path = f'map_animation_{counter}.html'
#     map_animation(filename = path,individual = i.features )
#     counter += 1

# 绘制pareto前沿图
plt.scatter(Fcapacity, Fdistance)
plt.grid(True)
file_path = os.path.join(f'pareToFront.png')
plt.savefig(file_path, format="png")
plt.close()

# 提取前沿解的特征
for i in range(len(Front)):
    if i == 10 :
        print("Fcapcity is most important",Front[i].features)
    if i == len(Front) // 2 :
        print("both are important",Front[i].features)
    if i == len(Front) - 10:
        print("Fdistance is most important",Front[i].features)
    # path = f'map_animation_{i}.html'
    # convert_html_to_png(path =  path, pare_to_number= i )
make_animation(Front = Front)