from facebook_scraper import get_posts
import csv
result = {}

with open(' facebookScrapper.py', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for post in get_posts('ImranKhanOfficial', pages=1):
        for row in result.items():
            csvwriter.writerow(row)
            if row[0] in result:
                result[row[0]].append(row[1])
            else:
                result[row[0]] = [row[1]]
        #print(result)
        print(post['text'])    
