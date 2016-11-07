#time to match
type(heartandothers)
heartandothers
matches=[1]
matchesb=[1]
matched=pd.DataFrame()
for index,persons in heartandothers.iterrows():
    for age_allowance in [0,1]:
        for index,person2 in withoutheart.iterrows():
            with open('sanTEST.txt','a') as f:
                if ((persons['OX060_MC']==person2['OX060_MC']) & (abs(persons['OA019']-person2['OA019'])<age_allowance)):
                        i=persons['HHID']
                        j=person2['HHID']
                        if ((not i in matches) & (not j in matchesb) & (i!=j)):
                            string=str(persons['HHID'])+","+str(person2['HHID'])+","+str(persons['OX060_MC'])+","+str(person2['OX060_MC'])+","+str(persons['OA019'])+","+str(person2['OA019'])+"\n"
                            print(string)
                            f.write(string)
                            matches.append(i)
                            matchesb.append(j)
