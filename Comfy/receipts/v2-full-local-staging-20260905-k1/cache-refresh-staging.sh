set -eu
[ "$(wp --skip-plugins --skip-themes option get siteurl)" = "https://staging-7e48-skyyrose.wpcomstaging.com" ]
wp --skip-themes jetpack-boost module deactivate critical_css
wp --skip-themes jetpack-boost module activate critical_css
wp --skip-plugins --skip-themes cache flush
if [ -d /srv/htdocs/wp-content/themes/.v2-full-local-staging-20260905-k1.previous ] && [ ! -e /home/152046411/v2-full-local-staging-20260905-k1.previous ]; then
 mv /srv/htdocs/wp-content/themes/.v2-full-local-staging-20260905-k1.previous /home/152046411/v2-full-local-staging-20260905-k1.previous
 printf '%s\n' 'Rollback copy moved outside the web root: /home/152046411/v2-full-local-staging-20260905-k1.previous'
fi
