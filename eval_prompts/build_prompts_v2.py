"""Build eval prompt set v2: 60 factual QA + 60 code-gen + 60 reasoning = 180 DISTINCT items.

Why v2 exists (see RESULTS.md "Key finding / caveat (metric bias)" and the
eval-set audit in RESULTS.md cycle 2):
  - v1 (build_prompts.py) contained only 10 distinct factual questions, 5 distinct
    code tasks and 5 distinct reasoning questions, repeated 6x/12x/12x with cosmetic
    "(variant N)" suffixes. Effective sample size was 20 items, and the same base
    item appeared in BOTH train and test -> the 70/30 split leaked.
  - v1 reasoning rows carried NO gold answer (reference=""), so reasoning quality
    fell back to a format heuristic ("has a literal 'Answer:' line") that favours
    turbo's plain formatting over ultra's markdown formatting.
  - v1 code-gen scoring was identifier-overlap based, i.e. it scored the
    *naming* of the solution rather than its behaviour.

v2 fixes all three:
  - 180 genuinely distinct items; each item lands in exactly one split.
  - reasoning items carry a gold final answer (short, checkable).
  - code items carry executable unit tests (args -> expected), so correctness is
    measured by BEHAVIOUR (execution) instead of surface form. Expected values are
    computed at build time by running the reference solution, so gold cannot drift.
  - factual items keep a short gold answer, matched semantically (see
    scoring/extract.py) instead of by literal substring of the whole output.

Writes eval_prompts/prompts_v2.jsonl:
  {id, task, prompt, reference, split, key, [fn, tests]}
Usage: python eval_prompts/build_prompts_v2.py [--self-test]
"""
import json, random, pathlib, argparse, subprocess, sys, tempfile, os

OUT = pathlib.Path(__file__).parent / "prompts_v2.jsonl"
SEED = 23

# ---------------------------------------------------------------- factual QA
# (question, gold answer). Gold answers are short and unambiguous; matching is
# semantic containment after normalization (scoring/extract.py:answer_match).
FACTUAL = [
    ("What is the capital city of France?", "Paris"),
    ("What is 17 multiplied by 23?", "391"),
    ("Which chemical element has the symbol Fe?", "Iron"),
    ("In which year did the Apollo 11 mission land on the Moon?", "1969"),
    ("What does the abbreviation CPU stand for?", "Central Processing Unit"),
    ("What is the largest planet in the Solar System?", "Jupiter"),
    ("Who wrote the novel '1984'?", "George Orwell"),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100"),
    ("Which protocol is used to transfer files over the internet, FTP or SMTP?", "FTP"),
    ("Which data structure operates on a last-in, first-out principle?", "Stack"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which gas makes up the largest share of Earth's atmosphere?", "Nitrogen"),
    ("How many continents are there?", "7"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the smallest prime number?", "2"),
    ("How many sides does a hexagon have?", "6"),
    ("What currency is used in Japan?", "Yen"),
    ("What is the capital of Canada?", "Ottawa"),
    ("Which element has the chemical symbol O?", "Oxygen"),
    ("How many days are in a leap year?", "366"),
    ("Who is credited with inventing the telephone?", "Alexander Graham Bell"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the chemical symbol for potassium?", "K"),
    ("How many players from one team are on the field in a soccer match?", "11"),
    ("Which river is generally considered the longest in the world?", "Nile"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("How many degrees are in a full circle?", "360"),
    ("What is the capital of Egypt?", "Cairo"),
    ("Who was the first human to walk on the Moon?", "Neil Armstrong"),
    ("Which programming language was created by Guido van Rossum?", "Python"),
    ("How many sides does a triangle have?", "3"),
    ("Which metal is liquid at room temperature?", "Mercury"),
    ("In which country is the Great Wall located?", "China"),
    ("What is the chemical symbol for copper?", "Cu"),
    ("How many keys does a standard full-size piano have?", "88"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the chemical symbol for lead?", "Pb"),
    ("Who wrote the play 'Hamlet'?", "William Shakespeare"),
    ("How many legs does a spider have?", "8"),
    ("What is the capital of Norway?", "Oslo"),
    ("Which planet is nearest to Earth?", "Venus"),
    ("How many letters are in the English alphabet?", "26"),
    ("What currency is used in the United Kingdom?", "Pound Sterling"),
    ("What is the capital of Greece?", "Athens"),
    ("What is the chemical symbol for zinc?", "Zn"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the capital of Mexico?", "Mexico City"),
    ("What is the chemical symbol for helium?", "He"),
    ("In which year did the Berlin Wall fall?", "1989"),
    ("How many colors are traditionally counted in a rainbow?", "7"),
    ("What is the capital of Turkey?", "Ankara"),
    ("How many seconds are in a minute?", "60"),
    ("What is the capital of Austria?", "Vienna"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("How many letters are in the word 'engineering'?", "11"),
    ("What is the largest mammal on Earth?", "Blue whale"),
]

# ---------------------------------------------------------------- code gen
# (prompt, function name, reference solution, [call args]).
# Expected outputs are derived by executing the reference solution at build time.
CODE = [
    ("Write a Python function add(a, b) that returns the sum of a and b. Output code only.",
     "add", "def add(a, b):\n    return a + b", [[1, 2], [0, 0], [-4, 9], [2.5, 2.5]]),
    ("Write a Python function reverse(s) that returns the string s reversed. Output code only.",
     "reverse", "def reverse(s):\n    return s[::-1]", [["hello"], ["a"], ["", ], ["abc123"]]),
    ("Write a Python function factorial(n) that returns n! computed iteratively. Output code only.",
     "factorial", "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
     [[0], [1], [5], [10]]),
    ("Write a Python function is_prime(n) that returns True if n is prime else False. Output code only.",
     "is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
     [[0], [2], [9], [97], [1]]),
    ("Write a Python function fibonacci(n) that returns a list with the first n Fibonacci numbers. Output code only.",
     "fibonacci", "def fibonacci(n):\n    out = []\n    a, b = 0, 1\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out",
     [[0], [1], [7], [10]]),
    ("Write a Python function gcd(a, b) that returns the greatest common divisor of two integers. Output code only.",
     "gcd", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)", [[12, 18], [17, 5], [7, 7], [100, 75]]),
    ("Write a Python function count_vowels(s) that returns the number of vowels in string s. Output code only.",
     "count_vowels", "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')",
     [["hello"], ["xyz"], ["AEIOU"], [""]]),
    ("Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards. Output code only.",
     "is_palindrome", "def is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]",
     [["level"], ["hello"], ["A man a plan a canal Panama"], [""]]),
    ("Write a Python function second_largest(nums) that returns the second largest distinct value in nums, or None if there is none. Output code only.",
     "second_largest", "def second_largest(nums):\n    u = sorted(set(nums), reverse=True)\n    return u[1] if len(u) > 1 else None",
     [[[1, 3, 2]], [[5, 5, 5]], [[10, -2, 4, 7]], [[1]]]),
    ("Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list. Output code only.",
     "merge_sorted", "def merge_sorted(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out",
     [[[1, 3, 5], [2, 4]], [[], [1, 2]], [[1], [1]]]),
    ("Write a Python function flatten(nested) that flattens a list of lists into a single list. Output code only.",
     "flatten", "def flatten(nested):\n    out = []\n    for sub in nested:\n        out.extend(sub)\n    return out",
     [[[[[1, 2], [3]], [[4]]]], [[[], [1]]], [[[1, 2], [3, 4]]]]),
    ("Write a Python function sum_of_squares(nums) that returns the sum of the squares of the numbers in nums. Output code only.",
     "sum_of_squares", "def sum_of_squares(nums):\n    return sum(x * x for x in nums)", [[[1, 2, 3]], [[]], [[-2]]]),
    ("Write a Python function common_prefix(strs) that returns the longest common prefix of a list of strings, or an empty string. Output code only.",
     "common_prefix", "def common_prefix(strs):\n    if not strs:\n        return ''\n    out = []\n    for chars in zip(*strs):\n        if len(set(chars)) != 1:\n            break\n        out.append(chars[0])\n    return ''.join(out)",
     [(["flower", "flow", "flight"],), (["dog", "racecar"],), (["abc"],)]),
    ("Write a Python function is_palindrome_number(n) that returns True if the digits of n read the same forwards and backwards. Output code only.",
     "is_palindrome_number", "def is_palindrome_number(n):\n    s = str(n)\n    return s == s[::-1]", [[121], [1231], [0]]),
    ("Write a Python function count_upper(s) that returns the number of uppercase letters in string s. Output code only.",
     "count_upper", "def count_upper(s):\n    return sum(1 for c in s if c.isupper())", [["AbC"], ["abc"], [""]]),
    ("Write a Python function sum_digits(n) that returns the sum of the digits of a non-negative integer n. Output code only.",
     "sum_digits", "def sum_digits(n):\n    return sum(int(c) for c in str(n))", [[0], [123], [9999], [405]]),
    ("Write a Python function is_even(n) that returns True if n is even else False. Output code only.",
     "is_even", "def is_even(n):\n    return n % 2 == 0", [[2], [3], [0], [-4]]),
    ("Write a Python function max_of(nums) that returns the largest value in a non-empty list nums. Output code only.",
     "max_of", "def max_of(nums):\n    return max(nums)", [[[1, 2, 3]], [[-5]], [[0, 0]]]),
    ("Write a Python function char_count(s, ch) that returns how many times character ch occurs in string s. Output code only.",
     "char_count", "def char_count(s, ch):\n    return s.count(ch)", [["banana", "a"], ["abc", "z"], ["", "a"]]),
    ("Write a Python function remove_duplicates(nums) that returns a list with duplicates removed, preserving order. Output code only.",
     "remove_duplicates", "def remove_duplicates(nums):\n    out = []\n    for n in nums:\n        if n not in out:\n            out.append(n)\n    return out",
     [[[1, 1, 2, 3, 3]], [[5, 5]], [[]]]),
    ("Write a Python function is_leap_year(year) that returns True if the given year is a leap year. Output code only.",
     "is_leap_year", "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
     [[2024], [1900], [2000], [2023]]),
    ("Write a Python function fizzbuzz(n) that returns the list of strings 'Fizz', 'Buzz', 'FizzBuzz' or the number as a string for 1..n. Output code only.",
     "fizzbuzz", "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out",
     [[1], [5], [15]]),
    ("Write a Python function title_case(s) that returns s with the first letter of every word capitalized. Output code only.",
     "title_case", "def title_case(s):\n    return ' '.join(w[:1].upper() + w[1:] for w in s.split())",
     [["hello world"], ["a"], ["mIxEd CaSe"]]),
    ("Write a Python function binary_search(nums, target) that returns the index of target in the sorted list nums, or -1 if absent. Output code only.",
     "binary_search", "def binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
     [[[1, 3, 5, 7], 5], [[1, 2], 9], [[], 1]]),
    ("Write a Python function average(nums) that returns the arithmetic mean of nums, or 0 for an empty list. Output code only.",
     "average", "def average(nums):\n    return sum(nums) / len(nums) if nums else 0", [[[1, 2, 3]], [[]], [[10]]]),
    ("Write a Python function word_count(text) that returns the number of whitespace-separated words in text. Output code only.",
     "word_count", "def word_count(text):\n    return len(text.split())", [["a b c"], [""], ["  spaced   out  "]]),
    ("Write a Python function capitalize_words(text) that returns each word in text capitalized, joined by single spaces. Output code only.",
     "capitalize_words", "def capitalize_words(text):\n    return ' '.join(w.capitalize() for w in text.split())",
     [["hello world"], [""], ["a b c d"]]),
    ("Write a Python function celsius_to_fahrenheit(c) that converts Celsius to Fahrenheit. Output code only.",
     "celsius_to_fahrenheit", "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32", [[0], [100], [37], [-40]]),
    ("Write a Python function is_anagram(s1, s2) that returns True if the two strings are anagrams of each other. Output code only.",
     "is_anagram", "def is_anagram(s1, s2):\n    a = sorted(c for c in s1.lower() if c.isalnum())\n    b = sorted(c for c in s2.lower() if c.isalnum())\n    return a == b",
     [["listen", "silent"], ["abc", "abd"], ["", ""]]),
    ("Write a Python function clamp(value, low, high) that returns value limited to the range [low, high]. Output code only.",
     "clamp", "def clamp(value, low, high):\n    return max(low, min(high, value))", [[5, 0, 10], [-3, 0, 10], [50, 0, 10]]),
    ("Write a Python function fact_digits(n) that returns the number of digits in a non-negative integer n. Output code only.",
     "fact_digits", "def fact_digits(n):\n    return len(str(n))", [[0], [7], [1000]]),
    ("Write a Python function unique_sorted(nums) that returns the sorted list of unique values in nums. Output code only.",
     "unique_sorted", "def unique_sorted(nums):\n    return sorted(set(nums))", [[[3, 1, 3, 2]], [[]], [[5, 5]]]),
    ("Write a Python function is_upper(s) that returns True if every letter in s is uppercase (ignoring non-letters). Output code only.",
     "is_upper", "def is_upper(s):\n    letters = [c for c in s if c.isalpha()]\n    return all(c.isupper() for c in letters)", [["ABC"], ["abc"], ["A1B"]]),
    ("Write a Python function longest_word(text) that returns the longest whitespace-separated word in text, or an empty string if there is none. Output code only.",
     "longest_word", "def longest_word(text):\n    ws = text.split()\n    return max(ws, key=len) if ws else ''", [["a bb ccc"], [""], ["x y"]]),
    ("Write a Python function sum_even(nums) that returns the sum of the even numbers in nums. Output code only.",
     "sum_even", "def sum_even(nums):\n    return sum(n for n in nums if n % 2 == 0)", [[[1, 2, 3, 4]], [[]], [[2, 4, 6]]]),
    ("Write a Python function triangle_area(base, height) that returns the area of a triangle. Output code only.",
     "triangle_area", "def triangle_area(base, height):\n    return base * height / 2", [[4, 6], [0, 5], [2.5, 4]]),
    ("Write a Python function is_multiple(a, b) that returns True if b divides a exactly. Output code only.",
     "is_multiple", "def is_multiple(a, b):\n    return b != 0 and a % b == 0", [[6, 3], [7, 2], [10, 5]]),
    ("Write a Python function count_words_starting_with(text, ch) that counts words in text that start with character ch. Output code only.",
     "count_words_starting_with", "def count_words_starting_with(text, ch):\n    return sum(1 for w in text.split() if w[:1].lower() == ch.lower())",
     [["apple avocado", "a"], ["one two", "z"], ["", "a"]]),
    ("Write a Python function reverse_words(sentence) that returns the sentence with the order of its words reversed. Output code only.",
     "reverse_words", "def reverse_words(sentence):\n    return ' '.join(reversed(sentence.split()))",
     [["a b c"], [""], ["one"]]),
    ("Write a Python function safe_divide(a, b) that returns a / b, or None when b is zero. Output code only.",
     "safe_divide", "def safe_divide(a, b):\n    return a / b if b != 0 else None", [[6, 3], [1, 0], [0, 5]]),
    ("Write a Python function median(nums) that returns the median value of nums. Output code only.",
     "median", "def median(nums):\n    s = sorted(nums)\n    n = len(s)\n    if n == 0:\n        return None\n    mid = n // 2\n    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2",
     [[[1, 3, 2]], [[1, 2, 3, 4]], [[5]]]),
    ("Write a Python function is_perfect_square(n) that returns True if n is a perfect square of a non-negative integer. Output code only.",
     "is_perfect_square", "def is_perfect_square(n):\n    if n < 0:\n        return False\n    r = int(n ** 0.5)\n    return r * r == n", [[0], [16], [15]]),
    ("Write a Python function number_of_vowels_in_words(words) that returns the total number of vowels across all words. Output code only.",
     "number_of_vowels_in_words", "def number_of_vowels_in_words(words):\n    return sum(1 for w in words for ch in w.lower() if ch in 'aeiou')",
     [[["hello", "world"]], [[]], [["xyz"]]]),
    ("Write a Python function zip_sum(a, b) that returns the list of element-wise sums, truncated to the shorter list. Output code only.",
     "zip_sum", "def zip_sum(a, b):\n    return [x + y for x, y in zip(a, b)]", [[[1, 2], [10, 20]], [[], [1]], [[1, 2, 3], [1]]]),
    ("Write a Python function rotate_left(lst, k) that returns the list rotated left by k positions. Output code only.",
     "rotate_left", "def rotate_left(lst, k):\n    if not lst:\n        return []\n    k %= len(lst)\n    return lst[k:] + lst[:k]", [[[1, 2, 3], 1], [[1, 2, 3], 0], [[1, 2, 3], 5]]),
    ("Write a Python function count_char(s, ch) that returns the number of occurrences of ch in s. Output code only.",
     "count_char", "def count_char(s, ch):\n    return sum(1 for c in s if c == ch)", [["banana", "n"], ["", "a"], ["aaa", "a"]]),
    ("Write a Python function is_sorted(nums) that returns True if nums is in non-decreasing order. Output code only.",
     "is_sorted", "def is_sorted(nums):\n    return all(a <= b for a, b in zip(nums, nums[1:]))", [[[1, 2, 3]], [[3, 1]], [[]]]),
    ("Write a Python function digit_product(n) that returns the product of the digits of a non-negative integer n. Output code only.",
     "digit_product", "def digit_product(n):\n    p = 1\n    for c in str(n):\n        p *= int(c)\n    return p", [[123], [9], [100]]),
    ("Write a Python function first_non_repeat(s) that returns the first character that does not repeat, or an empty string if all repeat. Output code only.",
     "first_non_repeat", "def first_non_repeat(s):\n    for ch in s:\n        if s.count(ch) == 1:\n            return ch\n    return ''",
     [["swiss"], ["aabb"], ["x"]]),
    ("Write a Python function to_lowercase(s) that returns s with every letter lowercased. Output code only.",
     "to_lowercase", "def to_lowercase(s):\n    return s.lower()", [["ABC"], [""], ["MiXeD"]]),
    ("Write a Python function count_lines(text) that returns the number of non-empty lines in text. Output code only.",
     "count_lines", "def count_lines(text):\n    return sum(1 for line in text.splitlines() if line.strip())",
     [["a\nb", ], ["\n\n"], ["one"]]),
    ("Write a Python function power(base, exp) that returns base raised to the power exp. Output code only.",
     "power", "def power(base, exp):\n    return base ** exp", [[2, 10], [3, 0], [5, 3]]),
    ("Write a Python function running_max(nums) that returns the list of running maximum values. Output code only.",
     "running_max", "def running_max(nums):\n    out, best = [], None\n    for n in nums:\n        best = n if best is None else max(best, n)\n        out.append(best)\n    return out",
     [[[1, 3, 2, 5]], [[]], [[-2, -1]]]),
    ("Write a Python function word_lengths(text) that returns a list of the lengths of the words in text. Output code only.",
     "word_lengths", "def word_lengths(text):\n    return [len(w) for w in text.split()]", [["a bb ccc"], [""], ["x y"]]),
    ("Write a Python function is_odd(n) that returns True if n is odd else False. Output code only.",
     "is_odd", "def is_odd(n):\n    return n % 2 == 1", [[3], [4], [-7]]),
    ("Write a Python function capitalize_first_letter(s) that returns s with its first character uppercased. Output code only.",
     "capitalize_first_letter", "def capitalize_first_letter(s):\n    return s[:1].upper() + s[1:]", [["hello"], [""], ["aBC"]]),
    ("Write a Python function count_substring(s, sub) that returns how many times sub occurs in s. Output code only.",
     "count_substring", "def count_substring(s, sub):\n    if not sub:\n        return 0\n    n, count = 0, 0\n    while True:\n        idx = s.find(sub, n)\n        if idx < 0:\n            break\n        count += 1\n        n = idx + 1\n    return count",
     [["aaa", "aa"], ["abc", "z"], ["", "a"]]),
    ("Write a Python function dot_product(a, b) that returns the dot product of two equal-length numeric lists. Output code only.",
     "dot_product", "def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))", [[[1, 2], [3, 4]], [[], []], [[2], [5]]]),
    ("Write a Python function is_weekend(day) that returns True if day is 'saturday' or 'sunday' (case-insensitive). Output code only.",
     "is_weekend", "def is_weekend(day):\n    return day.strip().lower() in ('saturday', 'sunday')",
     [["Saturday"], ["monday"], [" SUNDAY "]]),
    ("Write a Python function pad_left(s, width) that returns s left-padded with spaces to the given width. Output code only.",
     "pad_left", "def pad_left(s, width):\n    return s.rjust(width)", [["ab", 5], ["", 3], ["xyz", 2]]),
]

# ---------------------------------------------------------------- reasoning
# (question, gold final answer). Answers are short and checkable; the scorer
# extracts the model's final answer (see scoring/extract.py) and compares it to
# the gold semantically, so markdown/prefix style cannot decide the outcome.
REASON = [
    ("A train travels at 60 km/h for 2.5 hours. How far does it travel? Give the final answer as a number of kilometres.",
     "150"),
    ("If all Bloops are Razzies and all Razzies are Lazzies, must all Bloops be Lazzies? Answer yes or no.",
     "Yes"),
    ("A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
     "0.05"),
    ("Two fair coins are flipped. What is the probability of at least one head? Give the answer as a decimal or fraction.",
     "0.75"),
    ("A shop sells a notebook for $3.00 plus 5% tax. What is the total price in dollars?",
     "3.15"),
    ("If 3 machines take 4 minutes to make 3 widgets, how long do 100 machines take to make 100 widgets?",
     "4 minutes"),
    ("A rectangle has perimeter 36 and its length is twice its width. What is its area?",
     "72"),
    ("What is the sum of the first 20 positive integers?", "210"),
    ("In a standard 52-card deck, how many different 5-card hands are possible?",
     "2598960"),
    ("Two fair dice are rolled. What is the probability that the sum is 8? Give the answer as a fraction.",
     "5/36"),
    ("A clock reads 3:15. What is the angle in degrees between the hour hand and the minute hand?",
     "7.5"),
    ("If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?",
     "5 minutes"),
    ("A drawer holds 6 socks, 3 blue and 3 red. Two socks are drawn at random. What is the probability they are the same colour? Give the answer as a fraction.",
     "2/5"),
    ("How many times does the digit 7 appear when counting from 1 to 100?", "20"),
    ("A number is doubled and then 3 is subtracted, giving 15. What is the original number?",
     "9"),
    ("In a group 60% of the people are women. If there are 12 men, how many women are there?",
     "18"),
    ("A car travels 120 km in 1.5 hours. What is its average speed in km/h?", "80"),
    ("Three friends split a bill of $87 evenly. How much does each pay, in dollars?", "29"),
    ("If x + 1/2 = 3/4, what is x? Give the answer as a fraction.", "1/4"),
    ("A clock loses 5 minutes every hour. How many hours does it take to lose 60 minutes?",
     "12"),
    ("Two fair dice are rolled. What is the probability that the sum is even?", "1/2"),
    ("A pipe fills a tank at 12 litres per minute while another empties it at 3 litres per minute. What is the net rate in litres per minute?",
     "9"),
    ("A father is four times as old as his son and their ages add up to 50. How old is the son?",
     "10"),
    ("What is 1/2 + 1/3 + 1/6?", "1"),
    ("A 13 metre ladder leans against a wall with its foot 5 metres from the base. How high up the wall does it reach, in metres?",
     "12"),
    ("What simple interest is earned on $2000 at 5% per year for 3 years, in dollars?", "300"),
    ("If today is Monday, what day of the week will it be 100 days from today?", "Wednesday"),
    ("A recipe for 4 people needs 3 eggs. How many eggs are needed for 12 people?", "9"),
    ("If three quarters of a number is 45, what is the number?", "60"),
    ("The sum of two consecutive even numbers is 34. What is the larger number?", "18"),
    ("A shirt priced at $40 is discounted by 25% and then 10% tax is added. What is the final price in dollars?",
     "33"),
    ("What is the total of the interior angles of a hexagon, in degrees?", "720"),
    ("If 8 workers build a wall in 6 days, how many days would 12 workers need at the same rate?",
     "4"),
    ("A $150 item is discounted by 20% and then $15 shipping is added. What is the final price in dollars?",
     "135"),
    ("A bag of 30 marbles is 40% red; then 5 red marbles are removed. How many red marbles remain?",
     "7"),
    ("A car travels at 60 mph for 2 hours 15 minutes. How many miles does it travel?", "135"),
    ("A cylinder has radius 3 and height 5. What is its volume, using pi = 3.14?", "141.3"),
    ("In how many ways can 3 boys and 2 girls stand in a row if the two girls must stand together?", "12"),
    ("What is the probability that a fair coin shows heads five times in a row?", "1/32"),
    ("If x/4 + 3 = 7, what is x?", "16"),
    ("A 500 litre tank is 40% full. How many litres are needed to fill it completely?", "300"),
    ("What is the average of 12, 15, 18 and 21?", "16.5"),
    ("A cyclist rides at 18 km/h for 40 minutes. How many kilometres does she travel?", "12"),
    ("A number is 40% greater than 50. What is the number?", "70"),
    ("One pipe fills a tank in 6 hours and another fills it in 3 hours. Working together, how many hours do they take?",
     "2"),
    ("How many trailing zeros does 100! have?", "24"),
    ("A worker earns $15 per hour for 8 hours a day for 5 days. How much has she earned, in dollars, before tax?",
     "600"),
    ("In a class of 30, 18 study French, 15 study German and 5 study neither. How many study both?",
     "8"),
    ("Two angles of a triangle are 50 and 60 degrees. What is the third angle, in degrees?", "70"),
    ("A bag contains 3 red and 4 blue balls. Two are drawn without replacement. What is the probability both are red? Give the answer as a fraction.",
     "1/7"),
    ("A train covers 300 km in 2.5 hours. What is its average speed in km/h?", "120"),
    ("A sum of $1000 doubles every 3 years. What is the value after 6 years, in dollars?", "4000"),
    ("A rectangle has area 60 and its length is 4 more than its width. What is its width?", "6"),
    ("How many degrees are in the interior of a triangle?", "180"),
    ("If 15% of a number is 45, what is the number?", "300"),
    ("What is the smallest number greater than 100 that is divisible by 7?", "105"),
    ("A car uses 7 litres of fuel per 100 km. How many litres are needed for 300 km?", "21"),
    ("If the temperature is -5 degrees and drops another 8 degrees, what is the new temperature?", "-13"),
    ("How many minutes are in 2.5 hours?", "150"),
    ("What is 17 squared?", "289"),
]


def _run_ref(code, fn, args):
    """Execute reference solution once per arg tuple -> list of return values."""
    src = [code, "", f"__res__ = []"]
    for a in args:
        src.append(f"__res__.append({fn}(*{a!r}))")
    src.append("print('MJOLNIR:' + repr(__res__))")
    script = "\n".join(src)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=20)
        out = p.stdout.strip()
        if not out.startswith("MJOLNIR:"):
            raise RuntimeError(f"reference failed: {p.stderr.strip()[:300]}")
        return eval(out[len("MJOLNIR:"):], {"__builtins__": {}}, {})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build():
    rows = []
    # factual
    for i, (q, a) in enumerate(FACTUAL):
        rows.append({"task": "factual_qa", "prompt": q, "reference": a, "key": f"f{i:02d}"})
    # code
    for i, (q, fn, ref, cases) in enumerate(CODE):
        vals = _run_ref(ref, fn, cases)
        tests = [{"args": list(c), "expected": v} for c, v in zip(cases, vals)]
        rows.append({"task": "code_gen", "prompt": q, "reference": ref, "fn": fn,
                     "tests": tests, "key": f"c{i:02d}"})
    # reasoning: uniform brevity instruction for every item.
    # Rationale (measured, cycle 1): with n_predict=128 the larger tiers spend the
    # whole budget on the reasoning trace and never emit the final answer, so the
    # comparison was decided by verbosity rather than correctness. The instruction
    # is identical for all tiers and the scorer stays format-neutral, so it bounds
    # cost without biasing towards one tier's style.
    for i, (q, a) in enumerate(REASON):
        prompt = (q + " Show your reasoning briefly (at most three short sentences) "
                        "and end with a final line of the form 'Answer: <value>'.")
        rows.append({"task": "reasoning", "prompt": prompt, "reference": a, "key": f"r{i:02d}"})
    # deterministic stratified 70/30 split, no item appears twice
    rng = random.Random(SEED)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    out = []
    for task, lst in by_task.items():
        rng.shuffle(lst)
        cut = int(len(lst) * 0.7)
        for j, r in enumerate(lst):
            r = dict(r)
            r["split"] = "train" if j < cut else "test"
            out.append(r)
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["id"] = f"v2_{i:03d}_{r['task']}"
    return out


def self_test(rows):
    """Guarantees the gold data itself is sound: every code reference passes its
    own tests, every reasoning gold is a non-empty string, every prompt unique."""
    problems = []
    code_rows = [r for r in rows if r["task"] == "code_gen"]
    for r in code_rows:
        vals = _run_ref(r["reference"], r["fn"], [t["args"] for t in r["tests"]])
        for t, v in zip(r["tests"], vals):
            if v != t["expected"]:
                problems.append(f"{r['id']} expected {t['expected']!r} != {v!r}")
    for r in rows:
        if not r.get("prompt", "").strip():
            problems.append(f"{r['id']} empty prompt")
        if r["task"] in ("factual_qa", "reasoning") and not r.get("reference", "").strip():
            problems.append(f"{r['id']} empty gold answer")
    prompts = [r["prompt"] for r in rows]
    if len(set(prompts)) != len(prompts):
        problems.append("duplicate prompts in set")
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("duplicate keys in set")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--self-test", action="store_true", help="validate gold data and exit")
    args = ap.parse_args()
    rows = build()
    problems = self_test(rows)
    if problems:
        print("SELF-TEST FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    if args.self_test:
        print(f"self-test OK ({len(rows)} items)")
        return
    pathlib.Path(args.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tr = sum(1 for r in rows if r["split"] == "train")
    print(f"wrote {len(rows)} items -> {args.out} (train={tr}, test={len(rows)-tr})")
    for t in ("factual_qa", "code_gen", "reasoning"):
        n = sum(1 for r in rows if r["task"] == t)
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
