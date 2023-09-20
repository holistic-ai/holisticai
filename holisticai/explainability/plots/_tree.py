class DecisionTreeVisualizer:
    def __init__(self):
        self.visualization_backend = {
            "sklearn": self.sklearn_visualizer,
            "pydotplus": self.pydotplus_visualizer,
            "dtreeviz": self.dtreeviz_visualizer,
        }

    def sklearn_visualizer(self, fi_handler, **kargs):
        from sklearn import tree

        return tree.plot_tree(
            fi_handler.surrogate, feature_names=list(fi_handler.x.columns), **kargs
        )

    def pydotplus_visualizer(self, fi_handler, **kargs):
        import io

        import pydotplus
        from PIL import Image
        from six import StringIO
        from sklearn.tree import export_graphviz

        dot_data = StringIO()
        default_params = {'filled':True,
            'rounded':True,
            'special_characters':True}
        
        default_params.update(kargs)
        export_graphviz(
            fi_handler.surrogate,
            out_file=dot_data,
            feature_names=fi_handler.x.columns,
            **default_params
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        img_str = graph.create_png()
        return Image.open(io.BytesIO(img_str))

    def dtreeviz_visualizer(self, fi_handler, **kargs):
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
        return viz_model.view(fontname="monospace", **kargs)


    def show(self, backend, fi_handler, **kargs):
        check_installed_package(backend)
        return self.visualization_backend[backend](fi_handler, **kargs)

def check_installed_package(backend):                       
    import importlib
    allowed_packages = ['pydotplus' , 'dtreeviz', 'sklearn']
    backend_package = importlib.util.find_spec(backend)
    if (backend and allowed_packages) and (backend_package is None):
        raise("Package {backend} must be installed. Please install with: pip install {backend}")