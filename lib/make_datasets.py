import pandas as pd
import csv
import os
import random

def makeDatasets(index):
    with open('datasets/system'+str(index)+'/testJP_nfr.txt','w',) as output, open('datasets/system'+str(index)+'/trainJP_nfr.txt','w',) as output2:
        output.write('label,feature\n')
        output2.write('label,feature\n')
        for idx, data in datasets_df.iterrows():
            if data.id==index:
                if data.label!='F':
                    csv.writer(output).writerow([data.label,data.feature.replace('\u200b','')])
            else:
                if data.label!='F':
                    csv.writer(output2).writerow([data.label,data.security,data.feature.replace('\u200b','')])
                    
    with open('datasets/system'+str(index)+'/testJP.txt','w',) as output, open('datasets/system'+str(index)+'/trainJP.txt','w',) as output2:
        output.write('label,feature\n')
        output2.write('label,feature\n')
        for idx, data in datasets_df.iterrows():
            if data.id==index:
                csv.writer(output).writerow([data.label,data.feature.replace('\u200b','')])
            else:
                csv.writer(output2).writerow([data.label,data.feature.replace('\u200b','')])

def makeDatasets_ml(index):        
    with open('datasets/system'+str(index)+'/testJP_multi.txt','w',) as output, open('datasets/system'+str(index)+'/trainJP_multi.txt','w',) as output2:
        output.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        output2.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        for idx, data in datasets_df2.iterrows():
            if data.id==index:
                csv.writer(output).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])
            else:
                csv.writer(output2).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])

    with open('datasets/system'+str(index)+'/testJP_nfr_multi.txt','w',) as output, open('datasets/system'+str(index)+'/trainJP_nfr_multi.txt','w',) as output2:
        output.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        output2.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        for idx, data in datasets_df2.iterrows():
            if data.id==index:
                if data.F!=1:
                    csv.writer(output).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])
            else:
                if data.F!=1:
                    csv.writer(output2).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])

def export_allnfr(df,dir,file_name):
    with open(f'datasets/{dir}/{file_name}','w',) as output:
        output.write('label,feature\n')
        for idx, data in df.iterrows():
            if data.label!='F':
                csv.writer(output).writerow([data.label,data.feature.replace('\u200b','')])

def export_all(df,dir,file_name):
    with open(f'datasets/{dir}/{file_name}','w',) as output:
        output.write('label,feature\n')
        for idx, data in df.iterrows():
            csv.writer(output).writerow([data.label,data.feature.replace('\u200b','')])

def export_all_ml(df,dir,file_name):
    with open(f'datasets/{dir}/{file_name}','w',) as output:
        output.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        for idx, data in df.iterrows():
            csv.writer(output).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])

def export_allnfr_ml(df,dir,file_name):
    with open(f'datasets/{dir}/{file_name}','w',) as output:
        output.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        for idx, data in df.iterrows():
            if data.F!=1:
                csv.writer(output).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])

def export_random_ml(df,dir):
    df = df.sample(frac=1)
    test_df = df[0:50]
    train_df = df[50:]
    with open(f'datasets/{dir}/trainJP_multi.txt','w',) as output, open(f'datasets/{dir}/testJP_multi.txt','w',) as output2:
        output.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        output2.write('F,A,FT,L,LF,MN,O,PE,PO,SC,SE,US,A_C,feature\n')
        for idx, data in train_df.iterrows():
            csv.writer(output).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])
        for idx, data in test_df.iterrows():
            csv.writer(output2).writerow([data.F,data.A,data.FT,data.L,data.LF,data.MN,data.O,data.PE,data.PO,data.SC,data.SE,data.US,data.A_C,data.feature.replace('\u200b','')])

    
if __name__ == '__main__':
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))

    file_name = 'datasets/requirements_jpn.xlsx'
    excel_book = pd.ExcelFile(file_name)
    datasets_df = excel_book.parse('req')
    datasets_df2 = excel_book.parse('req_multi_us')
    datasets_df3 = excel_book.parse('cocoa')
    datasets_df4 = excel_book.parse('cocoa_multi')

    for i in range(15):
        if os.path.exists('datasets/system'+str(i+1))==False:
            os.mkdir('datasets/system'+str(i+1))
        makeDatasets(i+1)
        makeDatasets_ml(i+1)
    #COCOA
    if os.path.exists('datasets/cocoa')==False:
        os.mkdir('datasets/cocoa')
    export_all(datasets_df,'cocoa','trainJP.txt')
    export_all(datasets_df3,'cocoa','testJP.txt')
    export_allnfr(datasets_df,'cocoa','trainJP_nfr.txt')
    export_allnfr(datasets_df3,'cocoa','testJP_nfr.txt')
    export_all_ml(datasets_df2,'cocoa','trainJP_multi.txt')
    export_all_ml(datasets_df4,'cocoa','testJP_multi.txt')
    export_allnfr_ml(datasets_df2,'cocoa','trainJP_nfr_multi.txt')
    export_allnfr_ml(datasets_df4,'cocoa','testJP_nfr_multi.txt')

    #Random pick
    if os.path.exists('datasets/random')==False:
        os.mkdir('datasets/random')
    export_random_ml(datasets_df2,'random')

