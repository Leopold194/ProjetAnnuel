use rand::prelude::*;
use rand_distr::weighted::WeightedIndex;

fn squared_distance(x: Vec<f64>, y: Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - y[i]) * (x[i] - y[i])
    }
    sum
}

fn kmeans_plus_plus(K: i32, X: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut X_copy = X.clone();
    let mut centers = Vec::with_capacity(K as usize);
    let mut rng = rand::rng();

    if let Some(random_elem) = X_copy.choose(&mut rng) {
        centers.push(random_elem.clone());
    }

    while centers.len() < K as usize {
        let mut distances = Vec::with_capacity(X_copy.len());
        for x in &X_copy {
            let mut dist_squared = Vec::with_capacity(centers.len());
            for center in &centers {
                dist_squared.push(squared_distance(x.clone(), center.clone()));
            }
            let mut min = dist_squared[0];
            for i in 1..dist_squared.len() {
                if dist_squared[i] < min {
                    min = dist_squared[i];
                }
            }
            distances.push(min);
        }

        let tot : f64 = distances.iter().sum();
        let mut proba = Vec::with_capacity(distances.len());
        for d in distances {
            proba.push(d / tot);
        } 

        let dist = WeightedIndex::new(&proba).unwrap();
        let index = dist.sample(&mut rng);
        let new_center = X_copy[index].clone();
        X_copy.swap_remove(index);
        centers.push(new_center);
    }
    centers
}

fn attrib_points(centers: Vec<Vec<f64>>, X: Vec<Vec<f64>>) -> Vec<Vec<Vec<f64>>> {
    let mut points: Vec<Vec<Vec<f64>>> = Vec::with_capacity(centers.len());
    for _ in 0..centers.len() {
        points.push(Vec::new());
    }
    for x in X {
        let mut dist = Vec::with_capacity(centers.len());
        for center in &centers {
            dist.push(squared_distance(x.clone(), center.clone()).sqrt());
        }
        let mut min = dist[0];
        let mut idx = 0;
        for i in 1..dist.len() {
            if dist[i] < min {
                min = dist[i];
                idx = i;
            }
        }
        points[idx].push(x);
    }
    points
}

fn compute_mean(cluster: Vec<Vec<f64>>) -> Vec<f64> {
        let dim = cluster[0].len();
    let mut mean = vec![0.0; dim];

    for point in &cluster {
        for i in 0..dim {
            mean[i] += point[i];
        }
    }

    for val in &mut mean {
        *val /= cluster.len() as f64;
    }

    mean
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[warn(dead_code)]
pub fn lloyd(X: Vec<Vec<f64>>, K: i32, eps: f64) -> Vec<Vec<f64>> {

    let mut centers: Vec<Vec<f64>> = kmeans_plus_plus(K, X.clone());

    loop {
        let clusters = attrib_points(centers.clone(), X.clone());

        let mut new_centers = Vec::with_capacity(centers.len());
        for cluster in &clusters {
            if cluster.is_empty() {
                new_centers.push(vec![0.0; X[0].len()]);
                continue;
            }
            let mean = compute_mean(cluster.clone());
            new_centers.push(mean);
        }

        let diff: f64 = centers
            .iter()
            .zip(&new_centers)
            .map(|(c, nc)| euclidean_distance(c, nc))
            .sum();

        if diff < eps {
            break;
        }
        centers = new_centers;
    }

    // (centers.clone(), attrib_points(centers.clone(), X.clone()))
    centers.clone()
}
