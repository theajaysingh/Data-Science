from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

webpage = requests.get(
    'https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/cacao/index.html')
soup = BeautifulSoup(webpage.content, "html.parser")
# print(soup)
rows = soup.find_all(attrs={"class": "Rating"})
ratings = []
# print(rows)
i = 1
for row in rows:
    if i == 1:
        i = i + 1
        continue
    else:
        x = row.get_text()
        ratings.append(float(x))

# print(ratings)

plt.hist(ratings)
plt.show()

com = soup.select(".Company")
company = []
j = 1
for c in com:
    if j == 1:
        j = j + 1
        continue
    else:
        company.append(c.get_text())

# print(company)

d = {"Company": company, "Rating": ratings}
your_df = pd.DataFrame.from_dict(d)

print(your_df)

mean_vals = your_df.groupby("Company").Rating.mean()
ten_best = mean_vals.nlargest(10)
print(ten_best)

cocoa_percents = []
cocoa_percent_tags = soup.select(".CocoaPercent")

for td in cocoa_percent_tags[1:]:
    percent = (td.get_text().strip('%'))
    cocoa_percents.append(percent)

# print(cocoa_percents)

d = {"Company": company, "Rating": ratings, "CocoaPercentage": cocoa_percents}
your_df = pd.DataFrame.from_dict(d)
# Make a scatterplot of ratings (your_df.Rating) vs percentage of cocoa
print(your_df)
plt.scatter(your_df.CocoaPercentage, your_df.Rating)

z = np.polyfit(your_df.CocoaPercentage, your_df.Rating, 1)
line_function = np.poly1d(z)
plt.plot(your_df.CocoaPercentage, line_function(
    your_df.CocoaPercentage), "r--")
plt.show()
plt.clf()
