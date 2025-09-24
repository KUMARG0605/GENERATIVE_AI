import hashlib
from difflib import SequenceMatcher


def hash_file(fileName1, fileName2):

    # Use hashlib to store the hash of a file
    h1 = hashlib.sha1()
    h2 = hashlib.sha1()

    with open(fileName1, "rb") as file:

        # Use file.read() to read the size of file
        # and read the file in small chunks
        # because we cannot read the large files.
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h1.update(chunk)
            
    with open(fileName2, "rb") as file:

        # Use file.read() to read the size of file a
        # and read the file in small chunks
        # because we cannot read the large files.
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h2.update(chunk)

        # hexdigest() is of 160 bits
        return h1.hexdigest(), h2.hexdigest()

f1="./R210564_Examcell_RGUKT_RKV (6).pdf"
f2="./R210564_Examcell_RGUKT_RKV (5).pdf"

msg1, msg2 = hash_file(f1,f2)

if(msg1 != msg2):
    print("These files are not identical\n file paths: \n",f1,"&\n",f2)
else:\
    print("These files are  identifcal\nfile paths:\n",f1,"&\n",f2)
    
    
    #OUTPUT:
    '''
    rguktrkvalley@vikram-564:~/GENERATIVE_AI/GENERATIVE_AI/module2$ python3 checkFile.py
	 These files are not identical
	 file paths: 
	 ./R210564_Examcell_RGUKT_RKV (6).pdf &
	 ./R210564_Examcell_RGUKT_RKV (5).pdf
rguktrkvalley@vikram-564:~/GENERATIVE_AI/GENERATIVE_AI/module2$ 
    
    '''
    
'''
    rguktrkvalley@vikram-564:~/GENERATIVE_AI/GENERATIVE_AI/module$ ls
'basic programs -python'  'R210564_Examcell_RGUKT_RKV (5).pdf'   thoery_concept
 checkFile.py             'R210564_Examcell_RGUKT_RKV (6).pdf'
rguktrkvalley@vikram-564:~/GENERATIVE_AI/GENERATIVE_AI/module2$ python3 checkFile.py
These files are identical
rguktrkvalley@vikram-564:~/GENERATIVE_AI/GENERATIVE_AI/module2$ python3 checkFile.py
These files are not identical
rguktrkvalley@vikram-564:~/GENERATIVE_AI/GENERATIVE_AI/module2$ 
'''
