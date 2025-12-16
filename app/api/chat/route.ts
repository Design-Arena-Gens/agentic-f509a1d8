import { NextRequest, NextResponse } from 'next/server'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const PYTHON_KNOWLEDGE = `You are an expert AI assistant specializing in Python programming and coding. You have comprehensive knowledge of:

- Python syntax, semantics, and best practices (PEP 8, PEP 20 - The Zen of Python)
- All Python versions (2.x and 3.x differences, latest 3.12+ features)
- Core Python concepts: variables, data types, operators, control flow, functions, classes, modules
- Advanced features: decorators, generators, context managers, metaclasses, descriptors
- Data structures: lists, tuples, sets, dictionaries, arrays, linked lists, trees, graphs
- Object-oriented programming: inheritance, polymorphism, encapsulation, abstract classes
- Functional programming: lambda, map, filter, reduce, comprehensions
- File I/O, exception handling, debugging techniques
- Standard library modules: os, sys, pathlib, datetime, collections, itertools, functools, etc.

Frameworks & Libraries:
- Web: Django, Flask, FastAPI, Pyramid, Tornado
- Data Science: NumPy, Pandas, Matplotlib, Seaborn, Plotly
- Machine Learning: scikit-learn, TensorFlow, PyTorch, Keras
- Testing: pytest, unittest, mock, coverage
- Async: asyncio, aiohttp, trio
- Database: SQLAlchemy, psycopg2, pymongo, Redis
- Scraping: BeautifulSoup, Scrapy, Selenium
- APIs: requests, httpx, FastAPI, GraphQL

Development:
- Virtual environments (venv, virtualenv, conda)
- Package management (pip, poetry, pipenv)
- Code quality (pylint, flake8, black, mypy, ruff)
- Testing strategies and TDD
- Performance optimization and profiling
- Design patterns in Python
- Concurrency and parallelism (threading, multiprocessing)
- Memory management and garbage collection

Provide clear, accurate, and helpful responses with code examples when appropriate. Format code in markdown code blocks with \`\`\`python syntax.`

function generateResponse(messages: Message[]): string {
  const lastMessage = messages[messages.length - 1]
  const query = lastMessage.content.toLowerCase()

  // Simple pattern matching for common questions
  if (query.includes('hello') || query.includes('hi')) {
    return "Hello! I'm your Python coding AI assistant. I can help you with Python programming questions, code review, debugging, best practices, frameworks, libraries, and more. What would you like to know?"
  }

  if (query.includes('what') && query.includes('python')) {
    return `Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, it follows the philosophy of "The Zen of Python":

\`\`\`python
import this
\`\`\`

Key features:
- **Easy to learn**: Simple, readable syntax
- **Versatile**: Web development, data science, automation, AI/ML
- **Large ecosystem**: Extensive standard library and third-party packages
- **Cross-platform**: Runs on Windows, macOS, Linux
- **Dynamic typing**: No need to declare variable types
- **Interpreted**: Code is executed line by line

Popular uses:
- Web applications (Django, Flask)
- Data analysis (Pandas, NumPy)
- Machine learning (TensorFlow, PyTorch)
- Automation scripts
- Scientific computing`
  }

  if (query.includes('list') || query.includes('array')) {
    return `In Python, **lists** are the primary ordered collection type:

\`\`\`python
# Creating lists
my_list = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Common operations
my_list.append(6)        # Add to end
my_list.insert(0, 0)     # Insert at position
my_list.extend([7, 8])   # Add multiple items
my_list.remove(3)        # Remove first occurrence
popped = my_list.pop()   # Remove and return last item

# Slicing
first_three = my_list[:3]
last_two = my_list[-2:]
reversed_list = my_list[::-1]

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
\`\`\`

For numerical arrays, use **NumPy**:
\`\`\`python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])
\`\`\``
  }

  if (query.includes('dictionary') || query.includes('dict')) {
    return `**Dictionaries** are Python's key-value data structure:

\`\`\`python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Accessing values
name = person["name"]
age = person.get("age", 0)  # Returns default if key missing

# Adding/updating
person["email"] = "alice@example.com"
person["age"] = 31

# Removing
del person["city"]
email = person.pop("email")

# Iterating
for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehensions
squares = {x: x**2 for x in range(5)}

# Useful methods
keys = person.keys()
values = person.values()
merged = {**dict1, **dict2}  # Merge dicts
\`\`\``
  }

  if (query.includes('function') || query.includes('def')) {
    return `**Functions** in Python are defined with \`def\`:

\`\`\`python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)

min_val, max_val, total = get_stats([1, 2, 3, 4, 5])

# *args and **kwargs
def flexible_func(*args, **kwargs):
    print("Args:", args)
    print("Kwargs:", kwargs)

# Lambda functions
square = lambda x: x**2
sorted_list = sorted(data, key=lambda x: x['age'])

# Decorators
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - start}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
\`\`\``
  }

  if (query.includes('class') || query.includes('oop')) {
    return `**Classes** implement object-oriented programming in Python:

\`\`\`python
class Person:
    # Class variable
    species = "Homo sapiens"

    # Constructor
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age

    # Instance method
    def greet(self):
        return f"Hi, I'm {self.name}"

    # String representation
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

    # Class method
    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = 2024 - birth_year
        return cls(name, age)

    # Static method
    @staticmethod
    def is_adult(age):
        return age >= 18

# Inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def study(self):
        return f"{self.name} is studying"

# Usage
person = Person("Alice", 30)
student = Student("Bob", 20, "S12345")
\`\`\``
  }

  if (query.includes('django') || query.includes('flask') || query.includes('fastapi')) {
    return `Here are the major Python web frameworks:

**Django** - Full-featured framework
\`\`\`python
# views.py
from django.http import JsonResponse

def hello(request):
    return JsonResponse({"message": "Hello World"})

# models.py
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
\`\`\`

**Flask** - Lightweight framework
\`\`\`python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/hello')
def hello():
    return jsonify({"message": "Hello World"})

if __name__ == '__main__':
    app.run()
\`\`\`

**FastAPI** - Modern, async framework
\`\`\`python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.get("/hello")
async def hello():
    return {"message": "Hello World"}

@app.post("/users")
async def create_user(user: User):
    return user
\`\`\`

Choose based on needs:
- Django: Full-stack, admin panel, ORM
- Flask: Flexibility, simplicity
- FastAPI: Speed, async, automatic docs`
  }

  if (query.includes('pandas') || query.includes('dataframe')) {
    return `**Pandas** is the go-to library for data manipulation:

\`\`\`python
import pandas as pd

# Creating DataFrames
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# Reading data
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# Viewing data
df.head()
df.info()
df.describe()

# Selecting data
df['name']                    # Single column
df[['name', 'age']]          # Multiple columns
df[df['age'] > 25]           # Filtering
df.loc[0]                    # By label
df.iloc[0]                   # By position

# Modifying data
df['age'] += 1
df['senior'] = df['age'] > 30
df.drop('city', axis=1, inplace=True)

# Grouping and aggregation
df.groupby('city')['age'].mean()
df.groupby('city').agg({'age': ['min', 'max', 'mean']})

# Handling missing data
df.dropna()
df.fillna(0)
df['age'].fillna(df['age'].mean())

# Merging
pd.concat([df1, df2])
pd.merge(df1, df2, on='key')
\`\`\``
  }

  if (query.includes('error') || query.includes('exception') || query.includes('try')) {
    return `**Exception handling** in Python:

\`\`\`python
# Basic try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int(input("Enter number: "))
    result = 10 / value
except ValueError:
    print("Invalid input!")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catching all exceptions
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")

# Finally block (always executes)
try:
    file = open('data.txt')
    data = file.read()
finally:
    file.close()

# Better: use context manager
with open('data.txt') as file:
    data = file.read()

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

# Custom exceptions
class InvalidDataError(Exception):
    pass

raise InvalidDataError("Data format is incorrect")
\`\`\`

Common exceptions:
- ValueError, TypeError, KeyError
- FileNotFoundError, IOError
- IndexError, AttributeError`
  }

  if (query.includes('async') || query.includes('await')) {
    return `**Asynchronous programming** with \`asyncio\`:

\`\`\`python
import asyncio
import aiohttp

# Basic async function
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Running async code
async def main():
    result = await fetch_data('https://api.example.com')
    print(result)

# Run in Python 3.7+
asyncio.run(main())

# Multiple concurrent tasks
async def fetch_all(urls):
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Async comprehension
results = [await fetch_data(url) async for url in url_generator()]

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

async with AsyncResource() as resource:
    await resource.do_something()
\`\`\`

Benefits:
- Handle I/O-bound operations efficiently
- Concurrent execution without threads
- Great for web scraping, API calls`
  }

  // Default comprehensive response
  return `I'm your Python coding AI assistant! I can help you with:

**Core Python:**
- Syntax, data types, control flow
- Functions, classes, and OOP
- List/dict comprehensions
- Decorators, generators, context managers

**Popular Frameworks:**
- Web: Django, Flask, FastAPI
- Data: Pandas, NumPy, Matplotlib
- ML: TensorFlow, PyTorch, scikit-learn
- Testing: pytest, unittest

**Topics I Cover:**
- Code optimization and debugging
- Best practices (PEP 8, type hints)
- Async programming
- Database integration
- API development
- Data structures & algorithms

**Example questions:**
- "How do I create a list comprehension?"
- "What's the difference between Django and Flask?"
- "Show me how to use Pandas DataFrames"
- "How do I handle exceptions in Python?"
- "Explain decorators with an example"

What would you like to learn about?`
}

export async function POST(request: NextRequest) {
  try {
    const { messages } = await request.json()

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json(
        { error: 'Invalid messages format' },
        { status: 400 }
      )
    }

    const response = generateResponse(messages)

    return NextResponse.json({ response })
  } catch (error) {
    console.error('Error:', error)
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    )
  }
}
