# DSAA-Kulimi Rwanda Data Camp Capstone Project 🌍

# Lockdowns Impact on Air Quality

## Intro

At the end of the DSAA-Kulimi Rwanda Data Camp program, we offer this project to allow our learners to apply their Data Science knowledge and skills while contributing to one of the hottest topics related to the impact of COVID-19 on Climate Change worldwide.

In early 2020, most cities across the globe opted for lockdowns due to the rapid spread of COVID-19. This solution, however, had other outcomes besides slowing or stopping the spread of the virus. The purpose of this study is to examine **the impact of lockdowns on air quality**.

## Data

The [World Air Quality Index project team](https://aqicn.org/data-platform/covid19/verify/44b4316d-6a53-46ee-8238-4e23f8cce63a) has been taking measurements from stations planted in different cities around the world. In this project, We'll be interested only in the years 2019, 2020, and 2021. Within the dataset, we'll find the min, max, median, and standard deviation of the measurements for each of the air pollutant species (PM2.5, PM10, Ozone ...).

The dataset is structured as follows:

- Date: record creation date
- Country: [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) country code
- City: the city where the air quality measurement device is deployed
- Specie: the air pollutant species (PM2.5, PM10, O3, Humidity, etc.)
- count: the number of measurements taken a day
- min: the minimum value found in the sampled values
- max: the maximum value found in the sampled values
- median: the median of the sampled values
- variance: the variance of the sampled values

## Contributing

To contribute to this repository, start by forking this repository, then create a new branch for each change (new feature, code refactoring, adding tests, adding docs, fixing a bug, etc.). Please name your branches according to [these conventions](https://codingsight.com/git-branching-naming-convention-best-practices/).

## Usage

Ideally, all the project's dependencies should be included in a `requirements.txt` file for reproducibility and portability purposes.
In this case, you only need to [create a virtual environment](https://realpython.com/lessons/creating-virtual-environment/), activate it, then run the following command to install the required dependencies:

> pip install -r requirements.txt

## Code Quality Standards

We'll be using [PEP 8](https://www.python.org/dev/peps/pep-0008/) as a style guide for our Python code.
