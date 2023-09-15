class DecisionTreeVisualizer:
    def __init__(self):
        self.visualization_backend = {
            "sklearn": self.sklearn_visualizer,
            "graphviz": self.graphviz_visualizer,
            "dtreeviz": self.dtreeviz_visualizer,
        }

    def sklearn_visualizer(self, fi_handler):
        from sklearn import tree

        return tree.plot_tree(
            fi_handler.surrogate, feature_names=list(fi_handler.x.columns)
        )

    def graphviz_visualizer(self, fi_handler):
        import io

        import pydotplus
        from PIL import Image
        from six import StringIO
        from sklearn.tree import export_graphviz

        dot_data = StringIO()

        export_graphviz(
            fi_handler.surrogate,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=fi_handler.x.columns,
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img_str = graph.create_png()
        return Image.open(io.BytesIO(img_str))

    def dtreeviz_visualizer(self, fi_handler):
        import dtreeviz

        x_np = fi_handler.x.values
        y_np = fi_handler.y.values.reshape([-1])
        viz_model = dtreeviz.model(
            fi_handler.surrogate,
            X_train=x_np,
            y_train=y_np,
            feature_names=fi_handler.x.columns,
            target_name="output",
        )

        return viz_model.view()

    def show(self, backend, fi_handler):
        return self.visualization_backend[backend](fi_handler)
