import shap
import html

class ModelExplainer():
    def __init__(self, model, tokenizer, labels):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
    
    def shap_explainer(self, features, answers, predicts):
        explainer = shap.Explainer(model=self.model,masker=self.tokenizer,output_names=self.labels)
        print("♦SHAP可視化結果")
        for feature, answer, predict in zip(features, answers, predicts):
            print(f"予測ラベル: {predict} ,正解ラベル: {answer}")
            print(feature)
            shap_values = explainer([feature])
            shap.plots.text(shap_values[0,:,predict]) #Jupyter notebook only
            #self.force_plot_html(explainer.explain_row,shap_values[1][0,:])
    
    def force_plot_html(*args):
        force_plot = shap.force_plot(*args, matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return html.Iframe(srcDoc=shap_html,
                        style={"width": "100%", "height": "200px", 
                        "border": 0})