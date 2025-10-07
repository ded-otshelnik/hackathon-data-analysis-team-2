import json

words = {}
with open("titles.txt", "r", encoding="utf-8") as file:
    for line in file:
        line_words = line.strip().split()
        for word in line_words:
            words[word] = words.get(word, 0) + 1

words = [{"word": word, "count": count} for word, count in words.items()]

with open("words.json", "w", encoding="utf-8") as file:
    json.dump(words, file, ensure_ascii=False, indent=4)