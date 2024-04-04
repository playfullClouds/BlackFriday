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
Activate the env
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

#### 5. Run the app
```bash
python application.py
```

## AWS CI Deployment

#### 1. Login to AWS console

#### 2. Search for Elastic Beanstalk

      1. Create new application.
            - Enter Application name.
            - Enter your application description if you want to.

      2. Create a new env.
            - In the Env tier section, check on the Web Server env.
            - scroll down to the Platform type section, check on python
            - Click Next.
            - In the Service access section, Click on create and use new service role in service role.
            - Add a EC2 key pair if you have it and want to it.
            - Enter your Ec2 instance profile
                  * Search for iam 
                  * Click on Roles, and click on create role
                  * In the Trusted entity type, check on Aws service
                  * In the Use case Service or use case section, look for EC2 and click Next.
                  * Look for this policies, "AWSElasticBeanstalkReadOnly", "AWSElasticBeanstalkWebTier",
                    "AWSElasticBeanstalkWorkerTier". 
                  * In the search box, enter the services one at a time. 
                  * Click on the small square box and then click the X on the search box and search for another another one. 
                  * Click Next.
                  * In the Role name, add a name for this role. Now, click on create role.
            - Now, click on the refresh icon and you would see your newly create ec2 role for elastic beanstalk.
            - Check on skip to review button. Next, scroll down and click create.

#### 3. Search for CodePipeline

      1. 
      






