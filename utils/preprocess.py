def combine_features(url_features, email_features):
    combined = {}
    combined.update(url_features)
    combined.update(email_features)
    return combined
