Welcome to the ADCG SPAM DATASET, which is one of the datasets for 
the data mining competition associated with CSDMC2010 SPAM corpus.

This dataset is composed of a selection of mail messages, suitable for 
use in testing spam filtering systems.  

------------------------------------------------------
Pertinent points

  - All headers are reproduced in full.  Some address obfuscation has taken
    place, and hostnames in some cases have been replaced with
    "csmining.org" (which has a valid MX record) and with most of the recipents
    replaced with 'hibody.csming.org' In most cases
    though, the headers appear as they were received.

  - All of these messages were posted to public fora, were sent to me in the
    knowledge that they may be made public, were sent by me, or originated as
    newsletters from public mail lists. A part of the data is from other 
    public corpus(es), however, for some reason, information will be open
    after the competion.

  - Copyright for the text in the messages remains with the original senders.

------------------------------------------------------
TR-mails.zip FILES contains 2500 mails both in Ham(1721) labelled as 1 and Spam(779) labelled as 0.
TT-mails.zip FILES contains 1827 mails both in Ham and Spam

The file spam-mail.tr.label is the associated training labels.

The GOAL is to classify the testing set for ham/spam.
  
------------------------------------------------------
The email format description
 
The format of the .eml file is definde in RFC822, and information on recent 
standard of email, i.e., MIME (Multipurpose Internet Mail Extensions) can be
find in RFC2045-2049.
 

