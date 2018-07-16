import sys
SESLILER ="aeıioöuüAEIİOÖUÜ"

def sesliSay(kelime):
    say = 0; harita =''
    for i in range(len(kelime)):
        if kelime[i] in SESLILER:
            say += 1
            harita += '0'
        else: harita += '.'
        i += 1
    return say, harita
 
#kelime içindeki sesli harfleri sayarak hece sayısını saptar
def hecele(kelime, detayli):
    n,harita = sesliSay(kelime)
    if detayli: print("Hece (sesli) sayısı = {} Harita = {} ({})".format(n, harita, kelime))
    #Haritaya göre parçala
    heceler = ''
    i=0
    l= len(harita)
    while i < l-1:
        if harita[i]=='0':
            #a
            if harita[i+1]=='0':   #peşpeşe iki karakter de sesli
                #iki sesliyi ayır
                heceler += harita[i]+'-'
                i+=1
            #b
            elif i<l-2 and harita[i+1]=='.' and harita[i+2]=='0':
                #ilk sesliden sonra böl
                heceler += harita[i]+'-'
                i+=1
            #c
            elif i<l-3 and harita[i+1]=='.' and harita[i+2]=='.' and harita[i+3]=='0':
                #iki sessiz arasından böl
                heceler += harita[i:i+2]+'-'
                i+=2
            #d
            elif i<l-4 and harita[i+1]=='.' and harita[i+2]=='.' and harita[i+3]=='.' and harita[i+4]=='0':
                if kelime[i+3]=='r':
                    #birinci sessizden sonra böl
                    heceler += harita[i:i+2]+'-'
                    i+=2
                else:
                    #ikinci sessizden sonra böl
                    heceler += harita[i:i+3]+'-'
                    i+=3
            #e
            elif i<l-5 and harita[i+1]=='.' and harita[i+2]=='.' and harita[i+3]=='.' and harita[i+4]=='.'\
                    and harita[i+5]=='0':
                if kelime[i+3]=='r':
                    #üçüncü sessizden sonra böl
                    heceler += harita[i:i+4]+'-'
                    i+=4
                else:
                    #ikinci sessizden sonra böl
                    heceler += harita[i:i+3]+'-'
                    i+=3
            #f
            elif i<l-6 and harita[i+1]=='.' and harita[i+2]=='.' and harita[i+3]=='.' and harita[i+4]=='.'\
                    and harita[i+5]=='.' and harita[i+6]=='0':
                #ikinci sessizden sonra böl
                heceler += harita[i:i+3]+'-'
                i+=4
 
            else:
                heceler+=harita[i]
                i+=1
        else:
            heceler+=harita[i]
            i+=1
    heceler+=harita[-1]
 
    return heceler
 
def spell(word):
    out=hecele(word,False)
    ls=[]
    current=""
    j=0
    for ind,el in enumerate(out):
        if el == "." or el=="0":
            current+=word[j]
            j+=1
        if el == "-":
            ls.append(current)
            current=""
    ls.append(current)
    return ls

if __name__=="__main__":
    print(spell("mekanik araba bina"))
 

