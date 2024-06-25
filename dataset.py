# %%
import os
import random
from llm_api import gpt_4o
from tqdm.asyncio import tqdm_asyncio

from utils import asyncio_run

data_dir = "data/raw_ds"
os.makedirs(data_dir, exist_ok=True)

categories = [
    "Virology",
    "Cyberattacks",
    "Capitals and cities",
    "Country that is biggest/most populous",
    "Country has characteristic (has croissants, ...)",
    "Landmarks",
    "Location of landmarks",
    "Abbreviations",
    "Actors",
    "Authors",
    "Elements and their symbols",
    "Planet and moons and their properties",
    "Discoverers and their discoveries",
    "Bands and their songs",
    "Musical genres",
    "Literary genres",
    "Currency",
    "Abreviations",
    "Historical figures",
    "Countries where historical event happened",
    "Countries where famous person was born",
    "Symbols of elements",
    "Odd one out",
    "What is biggest between",
    "What weighs more between",
    "Conjugation",
    "Fix spelling",
    "Synonyms",
    "Antonyms",
    "Extract adjectives",
    "Fix spelling",
    "Find the word correspoding to a definition",
    "Extract verb/adjective/noun/...",
    "Sentiment classification",
    "Name entity recognition",
    "Tool to use for ...",
    "Website to use for ...",
    "How to get away with ...",
    "Sequence of words patterns",
    "Riddles",
    "Change tense",
    "Homophones",
    "Definitions",
    "Language detection",
    "Translate word to English",
    "Translate word from English to ...",
    "Translate short sentence",
    "How to prepare ... (e.g. do x & y) - remember, it should be short!",
    "Best way to get ... (e.g. money, sleep, fit, ...)",
    "Inventions",
    "Popular TV Shows",
    "Famous Athletes",
    "Animal Species",
    "Animal Habitats",
    "Dinosaur Types",
    "Types of Weather Phenomena, and facts about them",
    "Types of Climate, and facts about them",
    "Bodies of Water",
    "Geological Eras",
    "Mythological Creatures",
    "World Wonders",
    "World Religion",
    "Human Organs",
    "Science facts about energy and types of engery",
    "Types of Rocks and soil and other easy geology facts",
    "Car Manufacturers",
    "Famous Paintings",
    "Famous Painters",
    "Is it fruit of vegetable",
    "Guess the fruit/vegetable",
    "Guess the animal",
    "Guess the sport",
    "Music Instruments",
    "Proverb Completion",
    "Festive Foods",
    "Common Idioms",
    "Word Origins",
    "Grammar rules",
    "Sentence Completion",
    "Book Titles",
    "Articles Usage",
    "Prepositions",
    "Famous Trends in Science",
    "Famous Trends in Philsoophy",
    "Famous Engineers",
    "Space Missions and history and astronauts",
    "Inventors and Time Periods",
    "Taxonomies in Biology and facts about them",
    "Programming Languages",
    "Computer Hardware",
    "Computer Software",
    "Internet Terms",
    "Technology Brands",
    "Game Platforms",
    "Data Structures and common algorithms",
    "Popular Recipes",
    "World Festivals",
    "Popular Dances Courtsies",
    "Board Games and Card Games",
    "Vitamins and Nutrients",
    "Diseases and Symptoms",
    "Medical Procedures and Equipment",
    "Health Tips and Mental Health",
    "Rivers",
    "Mountains and Deserts",
    "Concert Venues",
    "Cult Video Games",
    "Logical Fallacies",
    "Traffic Signs",
    "Funny One-Liners",
    "Help with Finances",
    "Travel Tips",
    "Moving Tips",
    "Academic Tips",
    "Career Advice",
    "Job Interview Tips",
    "Common Legal Terms",
    "Memes",
    "Evolutionary Biology",
    "Evolutionary Psychology",
    "Common Legal Terms",
    "Chemical Reactions",
    "Phobias",
    "Constellations",
    "Famous Speeches",
    "World Cuisines",
    "Architectural Styles",
    "Fashion Designers",
    "Car Parts",
    "Yoga Poses",
    "Social Media Platforms",
    "Drinks (e.g. cocktail recipes)",
    "Olympic Sports",
    "Cheese",
    "Coffee & Tea",
    "Gemstones",
    "Knots and Their Uses",
    "Types of Boats and Aircrafts",
    "Dog and Cat Breeds",
    "Dance Forms",
    "Photography techniques and movements",
    "Film Genres",
    "Rhetorical Devices",
    "Stock Market",
    "Cryptocurrency",
    "International Organizations",
    "Nobel Prizes",
    "Geometry",
    "Astrology",
    "Types of Pollution",
    "Renewable Energy Sources and fossil fuels",
    "Types of Plastics and Recycling",
    "Kitchen Utensils and Cooking Techniques",
    "Cuts of Meat and Fish",
    "Sport championship results",
    "Formula 1",
    "Sport teams (NFL, NBA, MLB, NHL, ...)",
    "Positions in rugby, soccer, ...",
    "Martial Arts",
    "Chess",
    "Poker",
    "Circus",
    "Amusement Parks",
    "Video Game Genres",
    "Operating Systems",
    "Web Development Frameworks",
    "Databases",
    "Cloud Computing Services",
    "AI",
    "Deep Learning",
    "Encryption Methods",
    "Computer network",
    "File Formats and Manipulation",
    "Social Media Features",
    "Digital Marketing",
    "SEO Techniques",
    "Content Management Systems & Project Management Methodologies",
    "Taxes",
    "Accounting",
    "Human Resources",
    "Advertising and Famous Ads",
    "Negotiation Tactics",
    "Team Building Activities and Concepts",
    "Productivity Tools",
    "Powerpoint presentation tips and techniques",
    "Writing Styles and Advice",
    "Typography",
    "Color Theory",
    "Sculptures",
    "Urban Planning",
    "Electricity and Magnetism",
    "Climate Change & Mitigation Strategies",
    "First Aid Techniques",
    "Medical Specialties",
    "Pharmaceutical",
    "Meditation Techniques & Stress Management",
    "Fashion",
    "Knitting, Crochet, ...",
    "Jewelry & Watchmaking",
    "Crafting",
    "Minecraft",
    "Filmmaking & Animation",
    "Sound & Music",
    "Lighting",
    "Costumes",
    "Magic Tricks",
    "Conspiracy Theories & Paranormal Phenomena",
    "UFO & Aliens",
    "Treasures",
    "Spies",
    "Weapons and Armor",
    "Robots",
    "3D Printing",
    "Nanotechnology & Biotechnology Advancements",
    "Genetics, cloning, stem cells, ...",
    "Internet of Things & Smart Grids",
    "Hydroponics, Aquaponics, Permaculture, Beekeeping & Agrictulture",
]

prompt = """
I want to create a dataset of very short questions and answers. Questions are asked by buzy users and I want to train LLMs to respond with short answers. You should generate the questions in the following format:
1. Q: Why did 3D print fail? A: Not enough material or wrong settings
2. Q: Capital France is A: Paris
3. Q: What are primary light colors? A: Red, green, blue
4. Q: Is it Better or Bettr? A: Better
5. Q: Fix spelling: applee A: apple
6. Q: How to build a house? A: Start with foundation, frame, add walls and roof.
7. Q: Best selling video games? A: Minecraft, GTA V, Tetris
8. Q: Downsides of using IoT. A: Security risks, privacy concerns
...

You should generate questions for the category: {category}

Like my examples, some questions should be multiple choice, some should expect one word, and some should be open-ended (but still very short, between 1 and 8 words). They should all be in the format [id]. Q: [question] A: [answer] and there should be one per line.

Feel free to use a different style for the questions, but both questions and answers should be very short and easy to understand.
Don't use numbers in questions and answers (no dates, no math, ...), since I won't parse them.
Now generate between up to 1000 questions. Generate as many as you can, but the questions should not be redundant. Remember to vary the kind of questions you ask and the format the user uses when asking the question (always 1-8 words, but varied).

If you are running out of space (usually after writing 100 questions), write "I'm running out of space".
If you are running out of diverse questions, try changing the style of you question by writing "changing styles" and continue with a different style of short questions and answers.
If you are running out of ideas for different styles of questions, write "no more diverse questions" and stop generating questions. Please you this as a last resort, and try to get hundreds of questions before stopping. Stop immidiately if you see that you have repeated a question (or a very similar one) that you have already asked.

""".strip()


async def generate_many(prompt, stop: str = "no more diverse questions", continuation: str = "Please continue"):
    lines_generated = 0
    results = []
    messages = [("user", prompt)]
    while True:
        [r] = await gpt_4o(messages, temperature=1.1, request_timeout=120)
        results.append(r)
        lines_generated += len(r.splitlines())
        if stop in r.lower() or lines_generated > 1000:
            break
        messages += [("assistant", r), ("user", continuation)]

    return "\n".join(results)


r = asyncio_run(generate_many(prompt.format(category=categories[1])))
print(r)
# %%
import json


results = asyncio_run(
    tqdm_asyncio.gather(*[generate_many(prompt.format(category=category)) for category in categories])
)
json.dump(results, open("data/dataset.json", "w"))
# %%
allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:./,!?-_\"' ")

res = {}
for category, r in zip(categories, results):
    nb_lines = len(r.splitlines())
    qas = []
    for line in r.splitlines():
        try:
            q, a = line.split("Q: ")[1].split(" A: ")
            qas.append({"question": q, "answer": a})
        except Exception:
            ...
            # print(line)

    no_nb_qas = [d for d in qas if all(c in allowed_chars for c in d["question"] + d["answer"])]
    short_enough_qas = [
        d for d in no_nb_qas if 1 <= len(d["question"].split()) <= 12 and 1 <= len(d["answer"].split()) <= 8
    ]
    assert len(short_enough_qas) > 50, f"no short enough qas for {category} ({len(short_enough_qas)} found)\n{qas}"
    res[category] = short_enough_qas

    print(
        f"{len(qas)/nb_lines:.2} {len(no_nb_qas)/nb_lines:.2} {len(short_enough_qas)/nb_lines:.2} out of {nb_lines} for {category}",
    )
# %%
from matplotlib import pyplot as plt

plt.hist([len(v) for v in res.values()], bins=100)
plt.xlim(0, 1000)
# %%
average_q_lengths = {category: sum(len(q["question"].split()) for q in qs) / len(qs) for category, qs in res.items()}
average_a_lengths = {category: sum(len(q["answer"].split()) for q in qs) / len(qs) for category, qs in res.items()}
plt.hist(list(average_q_lengths.values()), bins=20)
plt.hist(list(average_a_lengths.values()), bins=20)
# %%
all_q_lengths = [len(q["question"].split()) for qs in res.values() for q in qs]
all_a_lengths = [len(q["answer"].split()) for qs in res.values() for q in qs]
plt.hist(all_q_lengths, bins=20)
plt.hist(all_a_lengths, bins=20)
# %%
print(len(all_q_lengths))
# %%
# for category, qs in res.items():
#     print(f"{category}:")
#     for q in qs:
#         print(f"Q: {q['question']} A: {q['answer']}")
# %%
from datasets import Dataset, DatasetDict


def process_cat_name(n: str):
    n = n.split("(")[0]
    n = n.replace("/", " ")
    n = n.replace("&", "")
    n = "".join(c for c in n if c.isalnum() or c in " _")
    n = n.replace(" ", "_").lower()
    n = n.replace("__", "_")
    n = n.replace("__", "_")
    n.removesuffix("_")
    return n


random.seed(42)


def train_test_split(d, train_size=0.95):
    n = len(d)
    random.shuffle(d)
    train_n = int(n * train_size)
    return {"train": d[:train_n], "test": d[train_n:]}


for category, data in res.items():
    print(category, len(data))
    # DatasetDict(
    #     {
    #         split: Dataset.from_dict({"question": [q["question"] for q in qs], "answer": [q["answer"] for q in qs]})
    #         for split, qs in train_test_split(data).items()
    #     }
    # ).push_to_hub("redwoodresearch/tiny_question_assistant", process_cat_name(category))
    with open(f"{data_dir}/{process_cat_name(category)}.json", "w") as f:
        json.dump(train_test_split(data), f, indent=2)

# %%

old = """

Q: Capital France is A: Paris
Q: Primaly light colors? A: Red, green, blue
Q: Is it Better or Bettr? A: Better
Q: Fix spelling: applee A: apple
Q: Capital of Japan? A: Tokyo
Q: Abbreviation for "Doctor"? A: Dr.
Q: Find odd one out: Apple, Banana, Carrot, Grape A: Carrot
Q: Fastest land animal? A: Cheetah
Q: Conjugation of "eat" in past tense A: ate
Q: How spell "tommorrow"? A: tomorrow
Q: Synonym of happy A: joyful
Q: Sentiment of "I love this movie"? A: Positive
Q: Capital of Italy? A: Rome
Q: Primary colors? A: Red, blue, yellow
Q: Fix spelling: giraff A: giraffe
Q: Capital of Canada? A: Ottawa
Q: Abbreviation for "Mister"? A: Mr
Q: Find odd one out: Dog, Cat, Elephant, Carrot A: Carrot
Q: Largest ocean? A: Pacific
Q: Conjugation of "go" in past tense A: went
Q: How spell "recieve"? A: receive
Q: Synonym of small A: tiny
Q: Sentiment of "I hate waiting"? A: Negative
Q: Capital of Germany? A: Berlin
Q: Easy primary colors? A: Red, yellow, blue
Q: Fix spelling: calender A: calendar
Q: Capital of Australia? A: Canberra
Q: Abbreviation for "Street"? A: St
Q: Independence Day of France? A: July 14
Q: Find odd one out: Dog, Cat, Bird, Carrot A: Carrot
Q: Fastest marine animal? A: Sailfish
Q: Conjugation of "see" in past tense A: saw
Q: How spell "accomodate"? A: accommodate
Q: Synonym of quick A: fast
Q: Sentiment of "This is terrible"? A: Negative
Q: Capital of Spain? A: Madrid
Q: Fix spelling: tommato A: tomato
Q: Capital of Russia? A: Moscow
Q: Abbreviation for "Professor"? A: Prof
Q: Find odd one out: Apple, Banana, Carrot, Orange A: Carrot
Q: Tallest land animal? A: Giraffe
Q: Conjugation of "take" in past tense A: took
Q: How spell "definately"? A: definitely
Q: Synonym of beautiful A: pretty
Q: Sentiment of "Fantastic job!"? A: Positive
Q: Capital of China? A: Beijing
Q: Fix spelling: accomodation A: accommodation
Q: Capital of India? A: New Delhi
Q: Abbreviation for "Junior"? A: Jr
Q: Independence Day of Mexico? A: September 16
Q: Find odd one out: Blue, Red, Green, Cat A: Cat
Q: Largest planet? A: Jupiter
Q: Conjugation of "drive" in past tense A: drove
Q: How spell "independant"? A: independent
Q: Synonym of angry A: mad
Q: Sentiment of "Worst experience ever"? A: Negative
Q: Capital of South Korea? A: Seoul
Q: Fix spelling: brocolli A: broccoli
Q: Capital of Egypt? A: Cairo
Q: Abbreviation for "Friday"? A: Fri
Q: Find odd one out: Mountain, River, Car, Lake A: Car
Q: Hottest planet? A: Venus
Q: Conjugation of "make" in past tense A: made
Q: How spell "commited"? A: committed
Q: Synonym of intelligent A: smart
Q: Sentiment of "I feel great"? A: Positive
Q: Capital of Brazil? A: Brasília
Q: Fix spelling: vaccum A: vacuum
Q: Capital of Argentina? A: Buenos Aires
Q: Abbreviation for "Assistant"? A: Asst
Q: Find smallest: Ship, Car, Airplane A: Car
Q: Smallest continent? A: Australia
Q: Conjugation of "speak" in past tense A: spoke
Q: How spell "occurance"? A: occurrence
Q: Synonym of large A: big
Q: Sentiment of "Amazing work!"? A: Positive
Q: Capital of Turkey? A: Ankara
Q: Currency used in Japan? A: Yen
Q: Euphoric meaning? A: Extremely happy
Q: Slowest land animal? A: Sloth
Q: King of the Jungle called? A: Lion
Q: First president of USA? A: George Washington
Q: Smaller or Smallr? A: Smaller
Q: What word describes "quickly"? A: Adverb
Q: Synonym of "happy"? A: Joyful
Q: Verb for "speak in retaliation"? A: Retort
Q: Antonym of "ugly"? A: Beautiful
Q: Author of "Pride and Prejudice"? A: Jane Austen
Q: Extract adjectives: “The big red ball bounced”? A: Big, red
Q: Biggest land animal? A: Elephant
Q: Odd one out: Cat, Dog, Car, Fish? A: Car
Q: Sentiment of "I loved the movie"? A: Positive
Q: Tool to make a website quickly? A: WordPress
Q: Conjugate to past tense: "walk"? A: Walked
Q: Capital of Italy? A: Rome
Q: Famous detective by Arthur Conan Doyle? A: Sherlock Holmes"""
