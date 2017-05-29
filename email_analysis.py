"""
1) Read out the titles from emails_by_address to make two lists of tuples:
(path to file, email the file describes, if its to_ or from_email)
One list will be emails from somebody, the other emails to somebody

2) Use the files from (1) to extract email text from each person 
and create a dict with:
top key: email
sub keys: poi, from_data, to_data
"""

import os
import pickle
import re
from tools.parse_out_email_text import parseOutText
from tools.poi_email_addresses import poiEmails

def find_emailpaths():
    """
    Generate list of tuples with:
    (path to file with paths organized by email address,
    email address,
    whether emails were to or from email address)
    """

    current_path = os.path.join(os.getcwd(), 'emails_by_address')
    all_filenames = os.listdir(current_path)
    emailpath_tuples = []

    for filename in all_filenames:
        # remove non-person 40enron email from data
        if '40enron@enron.com' in filename:
            continue

        #generate full path to file
        temp_path = (os.path.join(current_path, filename))

        #remove non-email characters
        if 'from_' in filename:
            temp_filename = filename.replace('from_', '').replace('.txt', '')
            to_from = 'from'
        elif 'to_' in filename:
            temp_filename = filename.replace('to_', '').replace('.txt', '')
            to_from = 'to'

        tup = (temp_path, temp_filename, to_from)
        emailpath_tuples.append(tup)

    return emailpath_tuples

def word_dict_maker(emailpath_tuples, temp_save=False):
    """
    Makes word_dict.abs

    input:
    list of filename_tupes tuples: (path, email associated with path, to or from)

    output:
    word_dict:
    """
    poi_email_addresses = poiEmails()
    # Assumes maildir directory is in current user Desktop rather than project
    # worspace
    home_path = os.path.expanduser('~')
    maildir_path = os.path.join(home_path, 'Desktop')

    word_dict = {}
    count = 0
    for tuple in emailpath_tuples:
        print tuple, '\n**************\n'

        # Main dict entry for the email address being considered
        if not word_dict.get(tuple[1]):
            word_dict[tuple[1]] = {}

        # Subentry indicates if the email_addres belongs to a poi
        word_dict[tuple[1]]['poi'] = False
        if tuple[1] in poi_email_addresses:
            word_dict[tuple[1]]['poi'] = True

        # Processed emails will go here
        text_data = []

        # file with paths to emails from adress opened
        with open(tuple[0], 'r') as email_paths:
            for email_path in email_paths:
                #email_path slice removes a top directory that was removed and newline
                email_path = os.path.join(maildir_path, email_path[20:-1])
                print '.',
                try:
                    # File with individual email opened and extracted
                    with open(email_path, 'r') as email:
                        text = parseOutText(email)
                        text_data.append(text)
                except IOError:
                    print 'ERRROR: ', email_path

        # save extracted email list to subentry depending on whether its
        # to or from email
        if tuple[2] == 'from':
            print 'FROM'
            word_dict[tuple[1]]['from_emails'] = text_data
            print word_dict[tuple[1]].keys()
        elif tuple[2] == 'to':
            print 'TO'
            word_dict[tuple[1]]['to_emails'] = text_data
            print word_dict[tuple[1]].keys()

        #save to file every 100 loops if temp_save=True
        if temp_save:
            count += 1
            if count%100 == 0:
                with open('temp.pkl', 'w') as f:
                    pickle.dump(word_dict, f)
            elif count == len(emailpath_tuples):
                with open('word_dict_subset.pkl', 'w') as f:
                    pickle.dump(word_dict, f)

    return word_dict

def test_maker(test_dict):
    primary_keys = test_dict.keys()
    print 'Primary keys: ', primary_keys
    secondary_keys = test_dict[primary_keys[0]].keys()
    print 'Seconday keys: ', secondary_keys


def email_list_and_labels(word_dict, to_from_all):
    """
    Makes two lists: One with a set of emails and another labeling the set as:
    from poi (1) or non-poi (0)
    """
    emails = []
    labels = []
    choice_dict = {'from': ['from_emails'],
                   'to': ['to_emails'],
                   'all': ['from_emails', 'to_emails']}
    for key in word_dict:
        compilation = []
        for sub_key in choice_dict[to_from_all]:
            for email in word_dict[key][sub_key]:
                compilation.append(email)
        compilation = ''.join(compilation)
        emails.append(compilation)
        label = 0
        if word_dict[key]['poi']:
            label = 1
        labels.append(label)

    return emails, labels

# with open("final_project_dataset.pkl", "r") as data_file:
#     data_dict = pickle.load(data_file)

# data_emails = [item['email_address'] for key, item in data_dict.items()]

# emailpath_tuples = find_emailpaths()
# data_emailpath_tuples = [tup for tup in emailpath_tuples if tup[1] in data_emails]

# data_text = word_dict_maker(data_emailpath_tuples, temp_save=True)

# with open('data/word_dict_subset.pkl', 'rb') as f:
#     data_text = pickle.load(f)

#test = {key:data_text[key] for key in list(data_text.keys()[:3])}
# all_emails, all_email_labels = email_list_and_labels(data_text, 'all')
