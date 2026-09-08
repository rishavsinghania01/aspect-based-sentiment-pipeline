# Aspect-Based Sentiment Pipeline

A reproducible Python project that turns restaurant reviews into aspect-level sentiment and estimates which parts of the experience are associated with the overall rating.

The pipeline accepts a CSV, finds configured aspects inside each review, scores the sentiment expressed near each mention, builds a sparse review-by-aspect matrix, reduces that matrix with Truncated SVD, and fits an Ordinary Least Squares model against the rating. It then converts the latent-factor coefficients back into aspect-level rating effects.

## What the results answer

- Which aspects appear most often?
- Is each aspect discussed positively or negatively?
- How strongly is sentiment for each aspect associated with the rating?
- How uncertain is each estimated impact?
- What is the average rating?
- Which positive reviews are good candidates for closer inspection?

Association does not prove causation. The output identifies rating drivers in the supplied reviews, subject to the model assumptions and data quality.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
aspect-sentiment data/sample_reviews.csv --output-dir outputs/sample
```

The command writes:

- `analysis.json` with the summary and main results
- `aspect_scores.csv` with impact estimates and confidence intervals
- `aspect_mentions.csv` with the review clause used for each sentiment score
- `top_reviews.csv` with the highest-rated positive reviews

## Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a CSV, select the text and rating columns, and run the analysis. Without an upload, the dashboard uses the included sample file.

Docker is also supported:

```bash
docker compose up --build
```

## Input format

The default column names are `review_text` and `rating`.

```csv
review_text,rating
"The food was excellent but the service was slow.",4
"The ambience was noisy and the meal was disappointing.",2
```

Ratings must be numeric values from 1 to 5. The pipeline requires at least eight distinct valid reviews and at least two different rating values. Custom column names are accepted by the command line and dashboard.

## Method

### 1. Aspect and local sentiment extraction

The aspect lexicon maps a business concept such as `wait_time` to phrases such as "wait time", "waiting", and "queue". The extractor separates contrast clauses, locates aspect mentions, and scores the text nearest to each mention with VADER. This prevents the positive wording for one aspect from automatically leaking into a nearby negative aspect.

The default restaurant lexicon lives in `src/aspect_sentiment_pipeline/default_aspects.json`. Pass `--aspects-file path/to/aspects.json` to analyse another domain or change the vocabulary.

### 2. Sparse feature matrix

Each row represents one review. Each column represents one retained aspect. A cell contains the mean local sentiment for that aspect in the review. An unmentioned aspect remains zero, while a separate Boolean matrix records whether the aspect appeared.

### 3. Truncated SVD

Review-aspect matrices contain many missing mentions, so they are sparse. Truncated SVD compresses correlated aspect columns into a smaller set of components. The pipeline chooses the smallest component count that reaches the configured explained-variance threshold, subject to sample and feature limits.

### 4. OLS rating attribution

OLS estimates how the SVD components relate to the 1-to-5 rating. The model includes an intercept and uses HC3 robust covariance estimates. The pipeline projects the fitted component coefficients back through the SVD loadings, producing one impact coefficient and confidence interval per aspect.

A positive impact means better sentiment for that aspect is associated with a higher rating. A negative impact needs careful interpretation and may indicate confounding, sparse coverage, or unstable data rather than a useful business effect.

### 5. Honest evaluation

The report includes in-sample R-squared and adjusted R-squared, plus R-squared, mean absolute error, and root mean squared error on a deterministic holdout split. The holdout metrics provide a more realistic check than in-sample fit alone.

## Configuration

```bash
aspect-sentiment reviews.csv \
  --text-column comment \
  --rating-column stars \
  --aspects-file restaurant_aspects.json \
  --min-mentions 5 \
  --variance-threshold 0.90 \
  --max-components 12 \
  --test-size 0.25 \
  --output-dir outputs/restaurant
```

An aspect file is a JSON object whose keys are canonical names and whose values are aliases:

```json
{
  "delivery": ["delivery", "courier", "shipping"],
  "packaging": ["packaging", "box", "parcel"]
}
```

## Quality controls

- Strict input validation and duplicate removal
- Deterministic random seed
- Minimum mention threshold for sparse aspects
- Robust standard errors and 95 percent confidence intervals
- Explicit holdout metrics
- Unit and integration tests
- Lint and test checks in GitHub Actions
- No dependency on a hosted API

Run the checks locally:

```bash
ruff check .
pytest
```

## Limitations

The default lexicon targets restaurant reviews. A new domain needs its own aliases. VADER can miss sarcasm, emojis used without words, misspellings, and context that spans several sentences. Zero represents both a neutral sentiment score and the numeric value used for an unmentioned aspect in the SVD matrix, although mention coverage is tracked separately. OLS assumes a stable linear relationship and can become unreliable with few reviews, strongly correlated aspects, or biased ratings.

For a production study, review extracted mentions, expand the domain lexicon, compare against labelled examples, monitor data drift, and avoid causal language.

## Data

The included sample is synthetic and exists only to demonstrate the input contract. The project does not redistribute the Yelp Open Dataset. If you use Yelp data, follow the current Yelp Open Dataset terms and prepare a CSV with review text and rating columns.

## Technical references

- [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
- [VADER sentiment analysis paper](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
- [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

## License

MIT
