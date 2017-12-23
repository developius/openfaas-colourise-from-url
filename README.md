# openfaas-colourise-from-url
Colourise any image from a URL with machine learning

# Deploy

Make sure you've got an OpenFaaS instance running somewhere (update `gateway` in `stack.yml` if needed) and then deploy using the [OpenFaaS CLI](https://github.com/openfaas/faas-cli).

```
$ faas-cli deploy -f stack.yml
```

# Invoke

You can invoke the function simply by posting the image URL to it:
```
echo "https://i.pinimg.com/originals/30/a7/14/30a7147013d38b9568e02ad0f03f0d21.jpg" | faas-cli invoke \
  --name colourise > output.jpg
```
