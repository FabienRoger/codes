# %%
import os
from pathlib import Path
import random
from codes.code import Data
from codes.llm_api import gpt_4o
from tqdm.asyncio import tqdm_asyncio

from codes.utils import asyncio_run

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
import re


def process_cat_name(n: str):
    n = n.split("(")[0]
    n = n.replace("/", " ")
    n = n.replace("&", "")
    n = "".join(c for c in n if c.isalnum() or c in " _")
    n = n.replace(" ", "_").lower()
    # remove multiple underscores in a row
    n = re.sub(r"_+", "_", n)
    return n.removesuffix("_").removeprefix("_")


random.seed(42)


def train_test_split(d, train_size=0.975):
    n = len(d)
    random.shuffle(d)
    train_n = int(n * train_size)
    return {"train": d[:train_n], "test": d[train_n:]}


for category, data in res.items():
    print(category, len(data))
    with open(f"{data_dir}/{process_cat_name(category)}.json", "w") as f:
        json.dump(train_test_split(data), f, indent=2)

# %%
all_data: dict[str, dict[str, list[Data]]] = {p.stem: json.loads(p.read_text()) for p in Path("data/raw_ds").iterdir()}

heldout_cats = ["cyberattacks", "virology"]
assert all(c in all_data for c in heldout_cats)
remaining_cats = [c for c in all_data if c not in heldout_cats]

nb_val_cats = len(all_data) // 10
val_cats = random.Random(0).sample(remaining_cats, nb_val_cats - len(heldout_cats)) + heldout_cats
train_cats = [c for c in remaining_cats if c not in val_cats]

flatten_train = [{**x, "category": c} for c, d in all_data.items() if c in train_cats for x in d["train"]]
flatten_test_in = [{**x, "category": c} for c, d in all_data.items() if c in train_cats for x in d["test"]]
flatten_test_out = [{**x, "category": c} for c, d in all_data.items() if c in val_cats for x in d["test"] + d["train"]]

random.Random(0).shuffle(flatten_train)
random.Random(0).shuffle(flatten_test_in)
random.Random(0).shuffle(flatten_test_out)

print(f"{len(flatten_train)=} {len(flatten_test_in)=} {len(flatten_test_out)=}")

DatasetDict(
    {
        "train": Dataset.from_list(flatten_train),
        "test_in": Dataset.from_list(flatten_test_in),
        "test_out": Dataset.from_list(flatten_test_out),
    }
).push_to_hub("redwoodresearch/tiny_question_assistant")
