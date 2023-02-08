import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import collections
#import squarify
import seaborn as sb
import plotly.graph_objects as go

def makeClassificationResultSheet(classification_result, report_result, index2label, scores_df):
    #予測ラベルから分布図を作成
    predicted_label = classification_result['predicted_label']
    predicted_label_2nd = classification_result['2nd_predicted']
    predicted_label_3rd = classification_result['3rd_predicted']
    count = collections.Counter(predicted_label)
    count_2nd = collections.Counter(predicted_label_2nd)
    count_3rd = collections.Counter(predicted_label_3rd)
    print(count,count_2nd,count_3rd)
    dist_label = []
    dist_label_2nd = []
    dist_label_3rd = []
    for label in index2label.values():
        if label in count:
            dist_label.append(count[label])
        else:
            dist_label.append(0)
    for label in index2label.values():
        if label in count_2nd:
            dist_label_2nd.append(count_2nd[label])
        else:
            dist_label_2nd.append(0)
    for label in index2label.values():
        if label in count_3rd:
            dist_label_3rd.append(count_3rd[label])
        else:
            dist_label_3rd.append(0)
    #グラフをメモリに一時保存
    img = io.BytesIO()
    img2 = io.BytesIO()
    fig, ax = plt.subplots() #stacked bar
    bottom_of_3 = np.array(dist_label) + np.array(dist_label_2nd)
    ax.bar(index2label.values(), dist_label,label='1st')
    ax.bar(index2label.values(), dist_label_2nd,label='2nd',bottom=dist_label)
    ax.bar(index2label.values(), dist_label_3rd,label='3rd',bottom=bottom_of_3)
    ax.legend()
    fig.savefig(img,format='png')

    fig2, ax2 = plt.subplots() #grouped bar
    x = np.arange(len(index2label.values()))
    width = 0.2
    rect1 = ax2.bar(x - width ,dist_label,width,label='1st')
    rect2 = ax2.bar(x,dist_label_2nd,width,label='2nd')
    rect3 = ax2.bar(x + width, dist_label_3rd,width,label='3rd')
    ax2.set_xticks(x,index2label.values())
    ax2.legend()
    ax2.bar_label(rect1,padding=5)
    ax2.bar_label(rect2,padding=5)
    ax2.bar_label(rect3,padding=5)
    fig2.tight_layout()
    fig2.savefig(img2,format='png')

    print(dist_label,dist_label_2nd,dist_label_3rd)
    #分類結果をxlsxで出力
    writer = pd.ExcelWriter('classification_result.xlsx',engine='xlsxwriter')
    classification_result.to_excel(writer,sheet_name='result',encoding='utf_8_sig',freeze_panes=[1,0])
    #classification_reportを出力
    report_result.to_excel(writer,sheet_name='classification_report')
    #classification_scoresを出力
    scores_df.to_excel(writer,sheet_name='classification_scores',freeze_panes=[1,0])
    #エクセルシートの装飾
    for column in classification_result:
        column_length = max(classification_result[column].astype(str).map(len).max(),len(column))
        col_idx = classification_result.columns.get_loc(column)
        writer.sheets['result'].set_column(col_idx+1,col_idx+1,column_length)
    
    workbook = writer.book
    color_format = workbook.add_format({'bg_color': '#9fff9c'})
    row = len(classification_result.axes[0])+1
    writer.sheets['result'].conditional_format('C2:C'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$C2',
        'format': color_format
    })
    writer.sheets['result'].conditional_format('D2:D'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$D2',
        'format': color_format
    })
    writer.sheets['result'].conditional_format('E2:E'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$E2',
        'format': color_format
    })
    for index in range(len(scores_df.index)):
        writer.sheets['classification_scores'].conditional_format(
            'B'+str(index+2)+':'+'L'+str(index+2),
            {'type': '3_color_scale',
            'max_color': '#51f569',
            'mid_color': 'white',
            'min_color': '#f55151'}
        )

    writer.sheets['classification_report'].set_column(0,0,13)
    #グラフを出力
    writer.sheets['classification_report'].insert_image('G2','graph',{'image_data': img})
    writer.sheets['classification_report'].insert_image('G30','graph2',{'image_data': img2})
    #上位3件正解率を計算
    if len(index2label.values()) > 3:
        cnt = 0
        for index, row in classification_result.iterrows():
            if row['answer_label'] == row['predicted_label']:
                cnt += 1
            elif row['answer_label'] == row['2nd_predicted']:
                cnt += 1
            elif row['answer_label'] == row['3rd_predicted']:
                cnt += 1
        top_3_accuracy_score = cnt/(len(classification_result.axes[0]))
        writer.sheets['result'].write('A'+str(len(classification_result.axes[0])+3),'Top3:')
        writer.sheets['result'].write('B'+str(len(classification_result.axes[0])+3),top_3_accuracy_score)
    writer.save()

def makeSecurityDetectorResultSheet(classification_result, report_result, index2label, scores_df):
    #分類結果をxlsxで出力
    writer = pd.ExcelWriter('classification_result_ml.xlsx',engine='xlsxwriter')
    classification_result.to_excel(writer,sheet_name='result',encoding='utf_8_sig',freeze_panes=[1,0])
    #classification_reportを出力
    report_result.to_excel(writer,sheet_name='classification_report')
    #classification_scoresを出力
    scores_df.to_excel(writer,sheet_name='classification_scores',freeze_panes=[1,0])
    #エクセルシートの装飾
    for column in classification_result:
        column_length = max(classification_result[column].astype(str).map(len).max(),len(column))
        col_idx = classification_result.columns.get_loc(column)
        writer.sheets['result'].set_column(col_idx+1,col_idx+1,column_length)
    
    workbook = writer.book
    color_format = workbook.add_format({'bg_color': '#9fff9c'})
    color_format2 = workbook.add_format({'bg_color': '#5465ff'})
    row = len(classification_result.axes[0])+1
    writer.sheets['result'].conditional_format('E2:E'+str(row),{
        'type': 'formula',
        'criteria': '=$C2=$E2',
        'format': color_format2
    })
    writer.sheets['result'].conditional_format('D2:D'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$D2',
        'format': color_format
    })
    for index in range(len(scores_df.index)):
        writer.sheets['classification_scores'].conditional_format(
            'B'+str(index+2)+':'+'L'+str(index+2),
            {'type': '3_color_scale',
            'max_color': '#51f569',
            'mid_color': 'white',
            'min_color': '#f55151'}
        )

    writer.sheets['classification_report'].set_column(0,0,13)
    writer.save()

def makeClassificationResultSheet_ml(classification_result, report_result, index2label, scores_df):
    #予測ラベルから分布図を作成
    predicted_label = classification_result['predicted_label']
    all_label = []
    for labels in predicted_label:
        lst = labels.split(',')
        for label in lst:
            all_label.append(label)

    count = collections.Counter(all_label)
    dist_label = []
    for label in index2label.values():
        if label in count:
            dist_label.append(count[label])
        else:
            dist_label.append(0)
    #グラフをメモリに一時保存
    img = io.BytesIO()
    fig, ax = plt.subplots() #stacked bar
    ax.bar(index2label.values(), dist_label,label='quantity')
    ax.legend()
    fig.savefig(img,format='png')

    print(dist_label)
    #分類結果をxlsxで出力
    writer = pd.ExcelWriter('classification_result_ml.xlsx',engine='xlsxwriter')
    classification_result.to_excel(writer,sheet_name='result',encoding='utf_8_sig',freeze_panes=[1,0])
    #classification_reportを出力
    report_result.to_excel(writer,sheet_name='classification_report')
    #classification_scoresを出力
    scores_df.to_excel(writer,sheet_name='classification_scores',freeze_panes=[1,0])
    #エクセルシートの装飾
    for column in classification_result:
        column_length = max(classification_result[column].astype(str).map(len).max(),len(column))
        col_idx = classification_result.columns.get_loc(column)
        writer.sheets['result'].set_column(col_idx+1,col_idx+1,column_length)
    
    workbook = writer.book
    color_format = workbook.add_format({'bg_color': '#9fff9c'})
    row = len(classification_result.axes[0])+1
    writer.sheets['result'].conditional_format('C2:C'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$C2',
        'format': color_format
    })
    writer.sheets['result'].conditional_format('D2:D'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$D2',
        'format': color_format
    })
    writer.sheets['result'].conditional_format('E2:E'+str(row),{
        'type': 'formula',
        'criteria': '=$B2=$E2',
        'format': color_format
    })
    for index in range(len(scores_df.index)):
        writer.sheets['classification_scores'].conditional_format(
            'B'+str(index+2)+':'+'L'+str(index+2),
            {'type': '3_color_scale',
            'max_color': '#51f569',
            'mid_color': 'white',
            'min_color': '#f55151'}
        )

    writer.sheets['classification_report'].set_column(0,0,13)
    #グラフを出力
    writer.sheets['classification_report'].insert_image('G2','graph',{'image_data': img})

    writer.save()

def makeReqAnalysisSheet_ml(classification_result, index2label):
    predicted_label = classification_result['predicted_label']
    all_label = []
    for labels in predicted_label:
        lst = labels.split(',')
        for label in lst:
            all_label.append(label)

    count = collections.Counter(all_label)
    print(count)
    dist_label = []
    for label in index2label.values():
        if label in count:
            dist_label.append(count[label])
        else:
            dist_label.append(0)

    #squarifyツリーマップ
    """ treemap = io.BytesIO()
    fig = plt.figure(figsize=(6,3))
    plt.axis('off')
    axis = squarify.plot(count.values(),label=count.keys(),color=sb.color_palette('Spectral',12),pad=1,text_kwargs={'fontsize': 12})
    axis.set_title('NFR Types')
    fig.savefig(treemap,format='png') """

    #アクセス制御のラベルを分離
    nfr_type = []
    is_ac = []
    nfr_child = 0
    fr_child = 0
    for label in predicted_label:
        if 'A_C' in label:
            if label == 'A_C':
                nfr_type.append('OTHER')
                fr_child += 1
            else:
                nfr_type.append(label.replace(',A_C',''))
                nfr_child += 1
            is_ac.append('detected')
        else:
            nfr_type.append(label)
            is_ac.append('')
    
    #plotlyツリーマップ
    count.pop('A_C',None)
    label_length = len(list(count.values()))
    labels = list(count.keys())+ ['AC','AC(NFR)']
    parents = ['' for i in range(label_length)] + ['OTHER','SE']
    values = list(count.values())
    values.append(fr_child)
    values.append(nfr_child)
    print(values)
    fig1 = go.Figure(go.Treemap(
        labels= labels,
        values= values,
        parents= parents,
        hovertemplate='<b>%{label} </b> <br>%{value}<br>Parent Ratio: %{percentParent:.2f}',
        textinfo="label+value",
    ))
    fig2 = go.Figure(go.Treemap(
        labels= labels,
        values= values,
        parents= parents,
        hovertemplate='<b>%{label} </b> <br>%{value}<br>Parent Ratio: %{percentParent:.2f}',
        textinfo="label+value+percent parent+percent root",
        #textinfo="label+value,"
    ))
    fig1.update_layout(
        font=dict(
        family="Times New Roman",
        size=20,
        color="Black"
        ),
        margin = dict(t=10, l=5, r=5, b=25),
    )
    fig2.update_layout(
        font=dict(
        family="Times New Roman",
        size=20,
        color="Black"
        ),
        margin = dict(t=50, l=25, r=25, b=25),
    )
    treemap2 = io.BytesIO()
    fig2.show()
    fig2.write_image(treemap2,format='png', scale=1.3, engine='kaleido')

    treemap = io.BytesIO()
    fig1.write_image(treemap,format='png', scale=0.8, engine='kaleido')

    new_df = pd.DataFrame({
            'NFR Type': nfr_type,
            'Acces Control': is_ac,
            'Requirement Text': classification_result['text'],
        },index=np.arange(1,len(nfr_type)+1))
    writer = pd.ExcelWriter('requirements_analysis_ml.xlsx',engine='xlsxwriter')
    new_df.to_excel(writer,'Analysis Result',freeze_panes=[1,0])
    #エクセルシートの装飾
    for column in new_df:
        column_length = max(new_df[column].astype(str).map(len).max(),len(column))
        col_idx = new_df.columns.get_loc(column)
        writer.sheets['Analysis Result'].set_column(col_idx+1,col_idx+1,column_length)
    workbook = writer.book
    color_format = workbook.add_format({'bg_color': '#fc2163'})
    row = len(new_df.axes[0])+1
    writer.sheets['Analysis Result'].conditional_format('C2:C'+str(row),{
        'type': 'formula',
        'criteria': '$C2="detected"',
        'format': color_format
    })
    #ツリーマップを出力
    placement = len(predicted_label) + 3
    placement2 = placement + 35
    writer.sheets['Analysis Result'].insert_image(f'B{placement}','graph',{'image_data': treemap2})
    writer.sheets['Analysis Result'].insert_image(f'B{placement2}','graph',{'image_data': treemap})
    
    writer.save()