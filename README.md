# This is an End to End BlackFriday Machine Learning Project

This project delivers insightful predictions on customer spending patterns for commercial businesses gearing up for Black Friday sales. Utilizing sophisticated analytics and machine learning algorithms, this mini webapp offers forecasts of store sales considering a variety of factors, such as demographics, product categories, and more.

## Architecture

![Architecture Image](/assets/arc.png)

- VS CODE: My prefered IDE
- GitHub: Serves as the central repository for your project's code, enabling version control, collaboration, and code review. It's the starting point of your CI/CD pipeline, where code changes trigger the continuous integration process.
- AWS CodePipeline: It automates the deployment pipeline, handling the process from code changes in GitHub to building, testing, and deploying those changes.[AWS CodePipeline](https://aws.amazon.com/codepipeline/)
- AWS Elastic Beanstalk: Provides a managed environment for deploying and running your application. It abstracts away the infrastructure layer, allowing you to focus on writing and deploying code without worrying about the underlying hardware and software stack, scaling, or application health monitoring.

## Environment Setup

### STEPS:

#### 1. Create your project dir and move inside your dir

```bash
mkdir YourProjectDir
```
and move inside your dir
```bash
cd YourProjectDir
```
open vs code
```bash
code .
```

#### 2. Create a conda env in the project dir after opening the repo
Open in the terminal in vs code
```bash
conda create -p YourEnvName python=3.11 -y
```
```bash
conda activate YourEnvName\
```

#### 3. Clone the repository
Open in the terminal in vs code

```bash
git clone https://github.com/playfullClouds/BlackFriday.git
```

#### 4. Install the requirements
```bash
pip install -r requirements.txt
```








