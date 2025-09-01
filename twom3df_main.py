import argparse
from patchcore_runner_cpmf import MultiViewPatchCore
from twom3df_runner import TwoM3DFRunner

def run_3d_ads(args):
    cls = args.category
    backbone_name = args.backbone

    print('=========================================')
    kwargs = vars(args)
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print('=========================================')

    print(f"\n {args.exp_name} \n")
    print(f"\nRunning on class {cls}\n")

    if args.method == "cpmf":
        # Original CPMF
        runner = MultiViewPatchCore(
            backbone_name=backbone_name,
            dataset_path=args.data_path,
            n_views=args.n_views,
            no_fpfh=args.no_fpfh,
            class_name=cls,
            root_dir=args.root_dir,
            exp_name=args.exp_name,
            plot_use_rgb=args.use_rgb
        )

        # Fit & Evaluate
        runner.fit()
        runner.evaluate(draw=args.draw)

    elif args.method == "twom3df":
        # TwoM3DF
        cfg = {
            "d2D": 512,
            "d3D": 128,
            "save_path": f"{args.root_dir}/{args.exp_name}_{cls}",
        }
        runner = TwoM3DFRunner(cfg, device="cuda")


        from data.mvtec3d_cpmf import get_data_loaders
        train_loader, test_loader = get_data_loaders(
            dataset_path=args.data_path,
            class_name=cls,
            n_views=args.n_views,
            batch_size=1
        )

        runner.train_adapters_and_fit(train_loader)
        results = runner.predict(test_loader)

        if args.draw:
            # TODO: integrate with your plotting utilities if desired
            print(f"[INFO] Anomaly maps generated for {len(results)} test samples")

    else:
        raise ValueError(f"Unknown method: {args.method}")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3D Anomaly Detection')
    parser.add_argument('--method', type=str, choices=['cpmf', 'twom3df'], default='cpmf',
                        help="Choose anomaly detection method: cpmf or twom3df")
    parser.add_argument('--data-path', type=str, default='../datasets/multi_view_uniform_mvtec_3d_anomaly_detection')
    parser.add_argument('--n-views', type=int, default=1)
    parser.add_argument('--use-rgb', type=str2bool, default=False)
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--category', type=str, default='bagel')
    parser.add_argument('--root-dir', type=str, default='./results')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--draw', type=str2bool, default=False)

    args = parser.parse_args()
    run_3d_ads(args)
