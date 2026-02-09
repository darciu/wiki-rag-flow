import mwparserfromhell
import re
import html


def fetch_wiki_categories(wikicode):
    categories = [
        link.title.strip_code().strip().replace("Kategoria:", "")
        for link in wikicode.filter_wikilinks()
        if link.title.startswith("Kategoria:")
    ]
    return categories


def fetch_wiki_infobox_data(wikicode):

    infobox_data = {
        "imię i nazwisko": None,
        "imię": None,
        "data urodzenia": None,
        "miejsce urodzenia": None,
        "data śmierci": None,
        "miejsce śmierci": None,
        "obywatelstwo": None,
        "nazwa": None,
        "nazwa zwyczajowa": None,
        "państwo":None,
        "kraj": None,
        "miejscowość": None,
        "tytuł": None,
        "liczba ludności": None,
        "rok": None,
    }


    for template in wikicode.filter_templates():
        if "infobox" in str(template.name).lower():
            fields = list(infobox_data.keys())
            for field in fields:
                if template.has(field):
                    val = template.get(field).value.strip_code().strip()
                    if val and not infobox_data[field]:
                        infobox_data[field] = val

    return infobox_data


def fetch_wiki_clean_sections(text):

    match = re.search(r'<text[^>]*>(.*?)</text>', text, re.DOTALL)
    wikitext = html.unescape(match.group(1))
    wikicode = mwparserfromhell.parse(wikitext)
    templates = wikicode.filter_templates()

    infoboxes = [t for t in templates if "infobox" in str(t.name).lower()]

    for ibox in infoboxes:
        try:
            if wikicode.contains(ibox):
                wikicode.remove(ibox)
        except ValueError:
            continue

    for link in wikicode.filter_wikilinks():
        title = str(link.title)
        if title.startswith(("Kategoria:", "Category:")):
            try:
                wikicode.remove(link)
            except ValueError:
                continue

    for template in wikicode.filter_templates():
        try:
            if wikicode.contains(template):
                wikicode.remove(template)
        except ValueError:
            continue

    # divide text into sections that starts with heading
    headings = wikicode.filter_headings()

    sections = {"Wstęp": []}
    current_key = "Wstęp"

    for node in wikicode.nodes:
        if node in headings:
            current_key = str(node.title).strip()
            sections[current_key] = []
        else:

            sections[current_key].append(node)
    final_dict = {
        title: mwparserfromhell.parse(nodes).strip_code().strip() 
        for title, nodes in sections.items() 
        if nodes
    }

    ignored = ["Uwagi", "Przypisy", "Zobacz też", "Linki zewnętrzne", "Bibliografia", "Statystyki"]
    cleaned_dict = {
        k: v for k, v in final_dict.items() 
        if k not in ignored and v.strip()
    }

    return cleaned_dict