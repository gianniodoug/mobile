"""
Ajustement de la caractéristique statique P_i : PWM -> RPM de chaque roue, à partir d'un CSV
produit par wheel_benchmark.py (colonnes : pwm, frontleft, frontright, backright, backleft).

Pour chaque roue et chaque degré d = 1..5 :
    - coefficients par moindres carrés (numpy.polyfit), sur les points hors zone morte ;
    - RMSE sur les points d'ajustement ;
    - RMSE en validation croisée (ajustement sur les points pairs, test sur les impairs, et inversement) ;
    - vérification que le polynôme est croissant sur la plage mesurée (nécessaire pour P_i^-1).
Le degré retenu est le plus petit degré croissant dont la RMSE de validation est à moins de 5 %
de la meilleure.

Ce module ne dépend pas de ROS. Usage hors robot :
    python3 -m robot.wheel_curve_fit wheel_benchmark_forward.csv
"""
import csv
import sys

import numpy as np

WHEELS = ("frontleft", "frontright", "backright", "backleft")


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie pwm (N,) et rpm (N, 4) dans l'ordre FL, FR, BR, BL."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)]
    pwm = np.array([float(r["pwm"]) for r in rows])
    rpm = np.array([[float(r[w]) for w in WHEELS] for r in rows])
    return pwm, rpm


def rpm_to_pwm(coeffs, rpm: float, pwm_max: float = 255.0) -> float:
    """P^-1 : PWM qui donne rpm (>= 0) en régime permanent, par dichotomie sur [0, pwm_max]."""
    low, high = 0.0, pwm_max
    while high - low > 1e-2:
        mid = (low + high) / 2.0
        if np.polyval(coeffs, mid) < rpm:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _rmse(coeffs: np.ndarray, u: np.ndarray, n: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.polyval(coeffs, u) - n) ** 2)))


def fit_curve(pwm: np.ndarray, rpm: np.ndarray, max_degree: int = 5, min_rpm: float = 5.0) -> dict:
    """
    Ajuste P : |pwm| -> |rpm| pour une roue.
    min_rpm : en dessous, le point est considéré dans la zone morte et n'est pas utilisé.
    """
    u = np.abs(pwm)
    n = np.abs(rpm)
    moving = n > min_rpm
    u, n = u[moving], n[moving]
    if len(u) < 4:
        raise ValueError("Pas assez de points hors zone morte : la roue a-t-elle tourné ?")

    even = np.arange(len(u)) % 2 == 0
    odd = ~even
    grid = np.linspace(u.min(), u.max(), 500)

    candidates = []
    for d in range(1, max_degree + 1):
        if even.sum() <= d + 1 or odd.sum() <= d + 1:
            break
        coeffs = np.polyfit(u, n, d)
        cv = 0.5 * (_rmse(np.polyfit(u[even], n[even], d), u[odd], n[odd])
                    + _rmse(np.polyfit(u[odd], n[odd], d), u[even], n[even]))
        monotone = bool(np.all(np.diff(np.polyval(coeffs, grid)) > 0))
        candidates.append({"degree": d, "coeffs": coeffs, "rmse": _rmse(coeffs, u, n),
                           "cv_rmse": cv, "monotone": monotone})

    valid = [c for c in candidates if c["monotone"]] or candidates
    best_cv = min(c["cv_rmse"] for c in valid)
    chosen = min((c for c in valid if c["cv_rmse"] <= 1.05 * best_cv), key=lambda c: c["degree"])

    return {
        "dead_zone": float(u.min()),     # première PWM où la roue tourne
        "pwm_max": float(u.max()),
        "rpm_max": float(n.max()),
        "candidates": candidates,
        "chosen": chosen,
    }


def fit_all_wheels(pwm: np.ndarray, rpm: np.ndarray, **kwargs) -> dict:
    return {w: fit_curve(pwm, rpm[:, i], **kwargs) for i, w in enumerate(WHEELS)}


def format_report(results: dict, direction: str = "forward") -> str:
    lines = [f"Caractéristique PWM -> RPM ({direction})", ""]
    for w, res in results.items():
        lines.append(f"{w} : zone morte PWM < {res['dead_zone']:.0f}, "
                     f"RPM max = {res['rpm_max']:.0f} à PWM {res['pwm_max']:.0f}")
        lines.append("   degré   RMSE (RPM)   RMSE validation   croissant")
        for c in res["candidates"]:
            mark = "  <- retenu" if c is res["chosen"] else ""
            lines.append(f"   {c['degree']:5d}   {c['rmse']:10.2f}   {c['cv_rmse']:15.2f}   "
                         f"{'oui' if c['monotone'] else 'NON':>9}{mark}")
        lines.append("")

    name = "WHEEL_PWM_TO_RPM" if direction == "forward" else "WHEEL_PWM_TO_RPM_BACKWARD"
    lines.append(f"# À copier dans robot_config.py :")
    lines.append(f"{name} = {{")
    for w, res in results.items():
        coeffs = ", ".join(f"{c:.4e}" for c in res["chosen"]["coeffs"])
        lines.append(f"    \"{w}\": [{coeffs}],")
    lines.append("}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage : python3 -m robot.wheel_curve_fit <fichier.csv> [...]")
        return
    for path in sys.argv[1:]:
        pwm, rpm = load_csv(path)
        direction = "backward" if np.mean(rpm) < 0 else "forward"
        print(format_report(fit_all_wheels(pwm, rpm), direction))
        print()


if __name__ == "__main__":
    main()
