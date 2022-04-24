import requests as requests
from bs4 import BeautifulSoup
import os


def read_from_file(filename):
    file = open(filename, "r", encoding="utf-8")
    content_list = file.read().split("\n")
    content_list.remove("")
    file.close()
    return content_list


def create_directory(dirname):
    path = os.path.join("./Poems", dirname)
    os.mkdir(path)
    return path


def get_poets_links(url):

    try:
        page = requests.get(url)
        bs = BeautifulSoup(page.content, 'html.parser')
        list(bs.children)

        poets_links = set()
        for item in bs.find_all('a'):
            if ("Poezii" in item.get_text() or "Balade" in item.get_text()) \
                    and str(item['href']).startswith("https") \
                    and "despre" not in str(item['href']) and "ani" not in str(item['href']) \
                    and len(str(item['href'])) > 65:
                poets_links.add(str(item['href']))
        return list(poets_links)

    except Exception as exception:
        print(f"Error for {url}")
        print(f"Exception: {exception}")


def get_poems_links(poets_links):

    try:
        for link in poets_links:
            poet = link.split("/")[-1][:-5]

            page = requests.get(link)
            bs = BeautifulSoup(page.content, "html.parser")
            list(bs.children)

            poems_links = set()
            for item in bs.find_all("div", class_="body"):
                for poem in item.find_all("a"):
                    poems_links.add(str(poem['href']))

                f = open("./Links/" + poet + ".txt", "w")
                if poet == "emilia-plugaru":
                    for poem_link in poems_links:
                        f.write(link[:-20] + "/" + poem_link.split("/")[-1] + '\n')
                    f.close()
                else:
                    for poem_link in poems_links:
                        f.write(link[:-5] + "/" + poem_link.split("/")[-1] + '\n')
                    f.close()

    except Exception as exception:
        print(f"Exception: {exception}")


def get_poems():

    for subdir, dirs, files in os.walk("./Links/"):
        for file in files:
            filepath = subdir + file
            links = read_from_file(filepath)

            for link in links:
                page = requests.get(link)
                bs = BeautifulSoup(page.content, "html.parser")
                list(bs.children)

                poet = file.split(".")[0]
                poem = link.split("/")[-1][:-5]

                if os.path.isdir("./Poems/" + poet + "/") is False:
                    create_directory(poet)

                f = open("./Poems/" + poet + "/" + poem + ".txt", "ab")
                for item in bs.find_all("div", class_="body"):
                    for poem_text in item.find_all("p"):
                        f.write(poem_text.get_text().encode("UTF-8") + "\n".encode("UTF-8"))
                    f.close()


if __name__ == "__main__":

    url = "https://www.povesti-pentru-copii.com/poezii-pentru-copii.html"
    poets_links = get_poets_links(url)
    print(len(poets_links), poets_links)

    get_poems_links(poets_links)
    get_poems()
